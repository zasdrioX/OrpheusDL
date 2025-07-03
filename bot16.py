import subprocess
import os
import glob
import time
import shutil
import zipfile
import requests
import humanize
import secrets
# --- CHANGE 1: MODIFIED IMPORT ---
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import ChatPermissions
from telegram.error import RetryAfter, TelegramError
from mutagen.flac import FLAC
from mutagen.mp3 import MP3, EasyMP3
from mutagen.wave import WAVE
from mutagen.mp4 import MP4
from mutagen.id3 import ID3
import sys
import json
import asyncio
import traceback
import re
import aiohttp
import certifi
import ssl
from typing import Dict, AsyncGenerator, List, Tuple
from functools import wraps

# Custom Exceptions
class PlaylistTooLargeError(Exception):
    pass

class DownloadCancelledError(Exception):
    pass

class RegionLockedError(Exception):
    pass

# Configuration
BOT_TOKEN = "6277779845:AAH2kjh9B0y-F8EVmPhYIOyLbQe9vFk3aKs"
ADMIN_ID = 731336143
TEMP_FOLDER = "temp_downloads"
DOWNLOAD_FOLDER = "downloads"
APPROVED_GROUPS_FILE = "approved_groups.json"
APPROVED_TOPICS_FILE = "approved_topics.json"
GOFILE_TOKEN = "wxWEuqKbUj1mZ2PTJMr49Ec03WrgwevV"
ORPHEUS_OVERALL_TIMEOUT_SECONDS = 3600 * 2
ORPHEUS_STALL_TIMEOUT_SECONDS = 600
ORPHEUS_READLINE_TIMEOUT_SECONDS = 180

# Global State for Queue System
download_queue = asyncio.Queue()
queue_lock = asyncio.Lock()
user_requests: Dict[int, Dict[str, str]] = {}
download_registry: Dict[str, Dict] = {}
download_tasks_lock = asyncio.Lock()


# Helper Functions
def escape_markdown_v2(text):
    return re.sub(r'([_*\\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def format_bytes(size_bytes):
    if size_bytes == 0: return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(abs(size_bytes).bit_length() - 1) // 10
    p = 1024 ** i
    s = round(size_bytes / p, 2)
    return f"{s}{size_name[i]}"

def format_speed(speed_bytes_per_sec):
    if speed_bytes_per_sec < 1024: return f"{speed_bytes_per_sec:.2f} B/s"
    return f"{format_bytes(int(speed_bytes_per_sec))}/s"

def parse_and_simplify_error(stderr_text: str) -> str:
    """Parses stderr and returns a user-friendly error message."""
    lower_stderr = stderr_text.lower()

    if any(kw in lower_stderr for kw in ["not found", "no result matching", "region-locked", "404", "error: album", "'nonetype' object has no attribute 'split'", "no release matches the given query"]):
        return "Not Found. Might be Region Restricted."

    if any(kw in lower_stderr for kw in ["login failed", "authentication error", "credentials", "could not log in"]):
        return "Bot authentication with the music service failed. Please notify the admin."

    if "quality not available" in lower_stderr:
        return "The requested download quality is not available for this item."

    lines = [line.strip() for line in stderr_text.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        cleaned_line = re.sub(r'^[a-zA-Z0-9._<>-]+:\s*', '', last_line).strip()
        if cleaned_line.lower().startswith("error:"):
            cleaned_line = cleaned_line[6:].strip()
        
        if cleaned_line:
            final_error = cleaned_line[0].upper() + cleaned_line[1:]
            return final_error[:250]

    return "An unknown error occurred during download."


# Monitoring Functions
async def monitor_download_progress(status_message, download_folder: str, platform_name: str, stop_event: asyncio.Event, progress_data: dict, dl_id: str):
    last_size = 0
    last_check_time = time.time()
    last_text = ""
    # --- CHANGE: Modified cancel message format ---
    cancel_info = f"\n\n> ❌ *Cancel:* /cancel\\_{dl_id}"

    while not stop_event.is_set():
        await asyncio.sleep(4)
        if not os.path.exists(download_folder): continue

        current_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(download_folder) for f in files)
        current_time = time.time()
        
        elapsed_time = current_time - last_check_time
        speed = (current_size - last_size) / elapsed_time if elapsed_time > 0 else 0
            
        last_size = current_size
        last_check_time = current_time
        
        track_progress_str = ""
        current_track = progress_data.get('current_track', 0)
        total_tracks = progress_data.get('total_tracks', 0)

        if total_tracks > 1:
            track_progress_str = f"Downloading Track: _{current_track}/{total_tracks}_\n"

        # --- CHANGE: Modified status_text to use new cancel_info without extra escaping ---
        status_text = (
            f"📥 *Downloading from {escape_markdown_v2(platform_name)}\\.\\.\\.*\n"
            f"{track_progress_str}"
            f"Downloaded: _{escape_markdown_v2(format_bytes(current_size))}_\n"
            f"Speed: _{escape_markdown_v2(format_speed(speed))}_"
            f"{cancel_info}"
        )
        try:
            if status_text != last_text:
                await status_message.edit_text(text=status_text, parse_mode='MarkdownV2')
                last_text = status_text
        except Exception as e:
            print(f"Failed to update download progress message: {e}")

async def monitor_upload_progress(status_message, progress: dict, stop_event: asyncio.Event, dl_id: str):
    last_check_time = time.time()
    last_bytes_sent = 0
    last_text = ""
    # --- CHANGE: Modified cancel message format ---
    cancel_info = f"\n\n> ❌ *Cancel:* /cancel\\_{dl_id}"

    while not stop_event.is_set():
        await asyncio.sleep(4)
        bytes_sent = progress.get("bytes_sent", 0)
        total_size = progress.get("total_size", 1)
        
        current_time = time.time()
        elapsed_time = current_time - last_check_time
        speed = (bytes_sent - last_bytes_sent) / elapsed_time if elapsed_time > 0 else 0

        last_bytes_sent = bytes_sent
        last_check_time = current_time
        
        percentage = (bytes_sent / total_size) * 100 if total_size > 0 else 0
        percentage_str = f"{percentage:.1f}"
        
        # --- CHANGE: Modified status_text to use new cancel_info without extra escaping ---
        status_text = (
            f"📤 *Uploading to GoFile\\.\\.\\.*\n"
            f"Uploaded: _{escape_markdown_v2(format_bytes(bytes_sent))} / {escape_markdown_v2(format_bytes(total_size))}_ "
            f"\\({escape_markdown_v2(percentage_str)}%\\)\n"
            f"Speed: _{escape_markdown_v2(format_speed(speed))}_"
            f"{cancel_info}"
        )
        try:
            if status_text != last_text:
                await status_message.edit_text(text=status_text, parse_mode='MarkdownV2')
                last_text = status_text
        except Exception as e:
            print(f"Failed to update upload progress message: {e}. Offending text: {status_text}")

# Queue Processor
async def queue_processor(app):
    print("Queue processor started")
    while True:
        item = await download_queue.get()
        (user_id, chat_id, message_thread_id, url,
         quality_tag, is_lossy_download_flag,
         reply_to_message_id, status_message, is_playlist, dl_id) = item

        async with download_tasks_lock:
            if dl_id not in download_registry:
                print(f"Task {dl_id} was cancelled in queue. Skipping.")
                download_queue.task_done()
                continue
            
            download_registry[dl_id]["status"] = "processing"
            cancel_event = download_registry[dl_id]["cancel_event"]

        try:
            await process_queue_item(
                app, user_id, chat_id, message_thread_id,
                url, quality_tag, is_lossy_download_flag,
                reply_to_message_id, status_message, is_playlist,
                dl_id, cancel_event
            )
        except Exception as e:
            print(f"Error processing queue item for {dl_id}: {e}")
            try:
                error_text = f"❌ Error: {escape_markdown_v2(str(e))}"
                await app.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=error_text,
                    parse_mode='MarkdownV2',
                    reply_to_message_id=reply_to_message_id
                )
            except Exception as send_error:
                print(f"Failed to send error message: {send_error}")
        finally:
            download_queue.task_done()
            async with queue_lock:
                if user_id in user_requests:
                    del user_requests[user_id]
            async with download_tasks_lock:
                if dl_id in download_registry:
                    del download_registry[dl_id]


async def process_queue_item(app, user_id, chat_id, message_thread_id, url,
                            quality_tag, is_lossy_download_flag,
                            reply_to_message_id, status_message, is_playlist,
                            dl_id, cancel_event):
    context = type('FakeContext', (), {'bot': app.bot})
    try:
        async with queue_lock:
            if user_id in user_requests:
                user_requests[user_id]["status"] = "processing"

        await download_url_implementation(
            user_id, chat_id, message_thread_id,
            url, quality_tag, is_lossy_download_flag,
            reply_to_message_id, context, status_message, is_playlist,
            dl_id, cancel_event
        )
    except Exception as e:
        print(f"Error in process_queue_item for {dl_id}: {e}")
        raise

# Download Implementation
async def download_url_implementation(user_id, chat_id, message_thread_id,
                                     url, quality_tag, is_lossy_download_flag,
                                     reply_to_message_id, context, status_message, is_playlist,
                                     dl_id, cancel_event):
    dl_monitor_task = None
    ul_monitor_task = None
    stop_dl_monitor_event = asyncio.Event()
    stop_ul_monitor_event = asyncio.Event()
    process = None

    try:
        if os.path.exists(DOWNLOAD_FOLDER): shutil.rmtree(DOWNLOAD_FOLDER)
        os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

        if not update_quality_config(quality_tag):
            await status_message.edit_text("❌ Failed to set download quality\\. Check logs\\.", parse_mode='MarkdownV2')
            return
        
        platform_name, platform_short = "Unknown", "U"
        for service_domain, p_name, p_short in [
            ("qobuz.com", "Qobuz", "Q"), ("deezer.com", "Deezer", "D"),
            ("tidal.com", "Tidal", "T"), ("jiosaavn.com", "JioSaavn", "J"),
            ("beatport.com", "Beatport", "B"), ("napster.com", "Napster", "N")]:
            if service_domain in url:
                platform_name, platform_short = p_name, p_short
                break
        
        progress_data = { "current_track": 0, "total_tracks": 0, "release_year_from_logs": None, "entity_title_from_logs": None,
                          "last_stdout_activity": time.time(), "last_stderr_activity": time.time(), 
                          "not_streamable_tracks": [], "last_track_num_seen": None, "invalid_url_detected": False }
        
        dl_monitor_task = asyncio.create_task(
            monitor_download_progress(status_message, DOWNLOAD_FOLDER, platform_name, stop_dl_monitor_event, progress_data, dl_id)
        )

        orpheus_start_time = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        orpheus_script_path = "orpheus.py"

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", orpheus_script_path, url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=".", env=env
        )
        async with download_tasks_lock:
            if dl_id in download_registry:
                download_registry[dl_id]["process"] = process

        stdout_lines, stderr_lines_live = [], []
        async def read_stream(stream, stream_name, lines_buffer, prog_data):
            while True:
                try:
                    line_bytes = await asyncio.wait_for(stream.readline(), timeout=ORPHEUS_READLINE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    if process.returncode is not None: break
                    continue
                except asyncio.CancelledError: break
                if not line_bytes: break
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if not line: continue
                lines_buffer.append(line)
                try: print(f"OrpheusDL ({stream_name}): {line}")
                except UnicodeEncodeError: print(f"OrpheusDL ({stream_name}): [line contains unprintable characters]")
                current_time_activity = time.time()
                if stream_name == 'stdout':
                    prog_data["last_stdout_activity"] = current_time_activity
                    
                    if "beatport.com" in url and "region locked" in line.lower():
                        raise RegionLockedError("Content is region locked.")
                        
                    if line.lower().startswith("title:"):
                        try: prog_data["entity_title_from_logs"] = line.split(":", 1)[1].strip()
                        except: pass
                    if "Release Date:" in line:
                        try: prog_data["release_year_from_logs"] = line.split(":")[-1].strip().split("-")[0].strip()
                        except: pass
                    
                    if line.strip().startswith("Invalid URL"):
                        prog_data["invalid_url_detected"] = True

                    if "Number of tracks:" in line:
                        try:
                            match = re.search(r'(\d+)', line)
                            if match:
                                total_tracks = int(match.group(1))
                                prog_data["total_tracks"] = total_tracks
                                if is_playlist and total_tracks > 50 and user_id != ADMIN_ID:
                                    raise PlaylistTooLargeError(f"Playlists are limited to 50 tracks. This one has {total_tracks}.")
                            else:
                                print(f"DEBUG: Could not parse number from 'Number of tracks' line: {line}")
                        except (ValueError, IndexError) as e:
                            print(f"DEBUG: Error parsing track number. Line: '{line}', Error: {e}")

                    if line.startswith("Track ") and "/" in line:
                        match = re.match(r'Track\s+(\d+)/\d+', line)
                        if match:
                            prog_data["last_track_num_seen"] = match.group(1)

                    if "=== Downloading track" in line:
                        prog_data["current_track"] += 1

                    if "not streamable!" in line.lower():
                        last_seen_num = prog_data.get("last_track_num_seen")
                        if last_seen_num:
                            not_streamable_list = prog_data.get("not_streamable_tracks", [])
                            if last_seen_num not in not_streamable_list:
                                not_streamable_list.append(last_seen_num)
                                prog_data["not_streamable_tracks"] = not_streamable_list
                else:
                    prog_data["last_stderr_activity"] = current_time_activity
        
        stdout_task = asyncio.create_task(read_stream(process.stdout, 'stdout', stdout_lines, progress_data))
        stderr_task = asyncio.create_task(read_stream(process.stderr, 'stderr', stderr_lines_live, progress_data))

        while process.returncode is None:
            if cancel_event.is_set():
                raise DownloadCancelledError(f"Download `{dl_id}` was cancelled by the user.")
            
            done, _ = await asyncio.wait([stdout_task, stderr_task], return_when=asyncio.FIRST_COMPLETED, timeout=1)
            for task in done:
                if task.exception():
                    raise task.exception()

            current_loop_time = time.time()
            if current_loop_time - orpheus_start_time > ORPHEUS_OVERALL_TIMEOUT_SECONDS:
                raise subprocess.TimeoutExpired(cmd=None, timeout=ORPHEUS_OVERALL_TIMEOUT_SECONDS, output="Overall timeout")
            last_activity = max(progress_data["last_stdout_activity"], progress_data["last_stderr_activity"])
            if current_loop_time - last_activity > ORPHEUS_STALL_TIMEOUT_SECONDS:
                raise subprocess.TimeoutExpired(cmd=None, timeout=ORPHEUS_STALL_TIMEOUT_SECONDS, output="Stall timeout")

        final_rc = await process.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        stop_dl_monitor_event.set()
        await dl_monitor_task

        if progress_data.get("invalid_url_detected"):
            await status_message.edit_text(text="❌ *Error:* Invalid URL", parse_mode='MarkdownV2')
            return

        if final_rc == 0:
            await status_message.edit_text(text=f"*Status:* _Download complete\\. Zipping files_\\.\\.", parse_mode='MarkdownV2')
            
            all_files_glob = glob.glob(os.path.join(DOWNLOAD_FOLDER, "**", "*.*"), recursive=True)
            audio_files_anywhere = [f for f in all_files_glob if any(f.lower().endswith(ext) for ext in [".flac", ".mp3", ".m4a", ".wav"])]
            not_streamable_tracks = sorted(progress_data.get("not_streamable_tracks", []), key=int)
            total_tracks = progress_data.get('total_tracks', 0)

            if not audio_files_anywhere:
                if total_tracks > 0 and len(not_streamable_tracks) == total_tracks:
                    await status_message.edit_text(text="❌ *Error:* Not Streamable", parse_mode='MarkdownV2')
                else:
                    await status_message.edit_text(text="❌ No audio files found post\\-download\\.", parse_mode='MarkdownV2')
                return

            audio_extensions = [".flac", ".mp3", ".m4a", ".ogg", ".wav"]
            image_extensions = [".png", ".jpg", ".jpeg"]
            files_to_zip = [p for p in all_files_glob if any(p.lower().endswith(ext) for ext in audio_extensions + image_extensions)]
            num_audio_zipped = len([f for f in files_to_zip if any(f.lower().endswith(ext) for ext in audio_extensions)])
            
            if num_audio_zipped == 0:
                await status_message.edit_text(text="❌ No audio files found to zip\\.", parse_mode='MarkdownV2')
                return
            
            is_single_track = num_audio_zipped == 1
            
            # --- Naming and Quality Logic ---
            zip_file_name_unsafe = "download.zip"

            if is_playlist:
                subdirs = [d for d in os.listdir(DOWNLOAD_FOLDER) if os.path.isdir(os.path.join(DOWNLOAD_FOLDER, d))]
                s_entity_name = sanitize_filename(subdirs[0]) if len(subdirs) == 1 else "Playlist"
                
                detected_formats = set()
                for f in audio_files_anywhere:
                    if is_dolby_atmos(f): 
                        detected_formats.add("ATMOS")
                        continue
                    ext = os.path.splitext(f)[1].lower()
                    if ext == '.flac': detected_formats.add('FLAC')
                    elif ext == '.mp3': detected_formats.add('MP3')
                    elif ext == '.m4a': detected_formats.add('AAC')
                
                format_order = {'ATMOS':-1, 'FLAC': 0, 'AAC': 1, 'MP3': 2}
                sorted_formats = sorted(list(detected_formats), key=lambda x: format_order.get(x, 99))
                
                quality_display_str = "/".join(sorted_formats) if sorted_formats else "N/A"
                zip_quality_str = quality_display_str

                platform_zip_part = f"({platform_short})"
                track_count_zip_part = f"[{num_audio_zipped}]"
                zip_file_name_unsafe = f"{s_entity_name} ({zip_quality_str}){platform_zip_part}{track_count_zip_part}.zip"

            else: # Logic for albums and single tracks
                is_atmos_track = is_dolby_atmos(audio_files_anywhere[0])
                release_year, album_from_metadata, album_artist_from_metadata, track_title_from_metadata = get_metadata(audio_files_anywhere[0])
                if progress_data["release_year_from_logs"]: release_year = progress_data["release_year_from_logs"]
                
                if is_atmos_track:
                    zip_quality_str = "ATMOS EAC3-JOC"
                    quality_display_str = "ATMOS EAC3-JOC"
                elif is_lossy_download_flag or "jiosaavn.com" in url:
                    _, _, bitrate_kbps = get_audio_metadata(audio_files_anywhere[0])
                    file_format_ext = os.path.splitext(audio_files_anywhere[0])[1].lower().replace('.', '')
                    display_format_ext = "AAC" if file_format_ext == "m4a" else file_format_ext.upper()
                    
                    if "jiosaavn.com" in url:
                        bitrate_disp = 320
                    else:
                        bitrate_disp = int(bitrate_kbps) if bitrate_kbps > 0 else 320
                    
                    zip_quality_str = f"{display_format_ext}-{bitrate_disp}K"
                    quality_display_str = f"{bitrate_disp}kbps {display_format_ext}"
                else: # Default lossless
                    bit_depth, sample_rate_khz, _ = get_audio_metadata(audio_files_anywhere[0])
                    file_format_ext = os.path.splitext(audio_files_anywhere[0])[1].lower().replace('.', '')
                    display_format_ext = "AAC" if file_format_ext == "m4a" else file_format_ext.upper()
                    sample_rate_disp = f"{int(sample_rate_khz)}" if sample_rate_khz.is_integer() else f"{sample_rate_khz:.1f}"
                    zip_quality_str = f"{display_format_ext}-{bit_depth}B-{sample_rate_disp}K"
                    quality_display_str = f"{bit_depth}bit-{sample_rate_disp}kHz {display_format_ext}"

                entity_name_for_file = album_from_metadata or (track_title_from_metadata if is_single_track else "Album")
                s_entity_name = sanitize_filename(entity_name_for_file)
                platform_zip_part = f"({platform_short})"

                if is_single_track:
                    s_track_title_display = sanitize_filename(track_title_from_metadata)
                    zip_file_name_unsafe = f"{s_track_title_display} ({zip_quality_str}){platform_zip_part}.zip"
                else: # Is an Album
                    year_zip_part = f" [{release_year}]" if release_year else ""
                    track_count_zip_part = f"[{num_audio_zipped}]"
                    zip_file_name_unsafe = f"{s_entity_name}{year_zip_part} ({zip_quality_str}){platform_zip_part}{track_count_zip_part}.zip"


            zip_path = create_zip(files_to_zip, zip_file_name_unsafe)
            if not zip_path:
                await status_message.edit_text(text="❌ Failed to create ZIP or no audio files zipped\\.", parse_mode='MarkdownV2')
                return

            zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            gofile_link = await upload_to_gofile(zip_path, context, status_message, dl_id, cancel_event)

            if gofile_link:
                platform_display = f"[{escape_markdown_v2(platform_name)}]({escape_markdown_v2(url)})"
                final_output_text = (
                    f"*Platform:* _{platform_display}_\n"
                    f"*File Size:* _{escape_markdown_v2(f'{zip_size_mb:.2f} MB')}_\n"
                    f"*Quality:* _{escape_markdown_v2(quality_display_str)}_\n"
                )

                if is_single_track:
                    album_display = escape_markdown_v2(album_from_metadata)
                    if release_year: album_display = escape_markdown_v2(f"{album_from_metadata} [{release_year}]")
                    final_output_text += f"*Album:* _{album_display}_\n"
                    
                    track_display = escape_markdown_v2(track_title_from_metadata)
                    final_output_text += f"*Track:* _{track_display}_\n"
                
                elif is_playlist:
                    playlist_display = escape_markdown_v2(s_entity_name)
                    final_output_text += f"*Playlist:* _{playlist_display}_\n"
                
                else: # Is an Album
                    album_display = escape_markdown_v2(s_entity_name)
                    if release_year: album_display = escape_markdown_v2(f"{s_entity_name} [{release_year}]")
                    final_output_text += f"*Album:* _{album_display}_\n"
                
                final_output_text += f"*Download Link:* [{escape_markdown_v2(os.path.basename(zip_path))}]({escape_markdown_v2(gofile_link)})"
                
                if not_streamable_tracks:
                    note = f"\\> _Note: Track\\(s\\) {', '.join(not_streamable_tracks)} were not streamable\\._"
                    final_output_text += f"\n\n{note}"

                await context.bot.send_message(
                    chat_id=chat_id, message_thread_id=message_thread_id,
                    text=final_output_text, parse_mode='MarkdownV2',
                    disable_web_page_preview=True, reply_to_message_id=reply_to_message_id
                )
                await status_message.delete()
            else:
                if not cancel_event.is_set():
                     await status_message.edit_text(text="❌ Failed to upload to GoFile\\. Check logs\\.", parse_mode='MarkdownV2')
            if os.path.exists(zip_path): os.remove(zip_path)
        else:
            full_stderr = "\n".join(stderr_lines_live)
            if full_stderr:
                simplified_error = parse_and_simplify_error(full_stderr)
            else:
                simplified_error = f"Download process failed with exit code {final_rc} but provided no error details."
            
            error_summary = f"❌ Error: {escape_markdown_v2(simplified_error)}"
            await status_message.edit_text(text=error_summary, parse_mode='MarkdownV2')

    except DownloadCancelledError as e:
        print(str(e))
        await status_message.edit_text(text=f"✅ _{escape_markdown_v2(str(e))}_", parse_mode='MarkdownV2')
    except PlaylistTooLargeError as e:
        await status_message.edit_text(text=f"❌ *Aborted:* _{escape_markdown_v2(str(e))}_", parse_mode='MarkdownV2')
    except RegionLockedError as e:
        await status_message.edit_text(text="❌ *Error:* Region Locked", parse_mode='MarkdownV2')
    except subprocess.TimeoutExpired as te:
        reason = str(te.output if te.output else "Operation timed out")
        await status_message.edit_text(text=f"❌ Download timed out: {escape_markdown_v2(reason)}\\.", parse_mode='MarkdownV2')
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).replace('-', '\\-')
        await status_message.edit_text(text=f"❌ Bot Error: {escape_markdown_v2(error_str)}\\. Check logs\\.", parse_mode='MarkdownV2')
    finally:
        if process and process.returncode is None: process.terminate()
        for task in [dl_monitor_task, ul_monitor_task]:
            if task and not task.done(): task.cancel()
        shutil.rmtree(DOWNLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)

# --- Upload Function ---
async def upload_to_gofile(file_path: str, context, status_message, dl_id: str, cancel_event: asyncio.Event):
    timeout_seconds = 900
    upload_url = "https://upload-na-phx.gofile.io/uploadFile"
    zip_size_bytes = os.path.getsize(file_path)
    upload_progress = {"bytes_sent": 0, "total_size": zip_size_bytes}
    stop_ul_monitor_event = asyncio.Event()
    ul_monitor_task = asyncio.create_task(
        monitor_upload_progress(status_message, upload_progress, stop_ul_monitor_event, dl_id)
    )
    try:
        async def file_sender_with_progress(fp: str, prog: dict, stop_event: asyncio.Event):
            chunk_size = 65536
            with open(fp, 'rb') as f:
                while True:
                    if stop_event.is_set():
                        raise DownloadCancelledError(f"Upload `{dl_id}` was cancelled by the user.")
                    chunk = f.read(chunk_size)
                    if not chunk: break
                    prog["bytes_sent"] += len(chunk)
                    yield chunk
                    await asyncio.sleep(0)  # Yield control to event loop

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        for attempt in range(3):
            if cancel_event.is_set(): raise DownloadCancelledError(f"Upload `{dl_id}` was cancelled by the user.")
            try:
                with aiohttp.MultipartWriter('form-data') as writer:
                    writer.append_form([('token', GOFILE_TOKEN)])
                    payload = aiohttp.payload.get_payload(
                        file_sender_with_progress(file_path, upload_progress, cancel_event),
                        content_type='application/zip',
                        headers={'Content-Disposition': f'form-data; name="file"; filename="{sanitize_filename(os.path.basename(file_path))}"'}
                    )
                    writer.append_payload(payload)
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as session:
                        async with session.post(upload_url, data=writer, ssl=ssl_context) as response:
                            response.raise_for_status()
                            json_response = await response.json()
                link = json_response.get('data', {}).get('downloadPage')
                if link: return link
            except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
                if attempt == 2: raise e
                await asyncio.sleep(10)
    except DownloadCancelledError:
        raise # Re-raise to be caught by the main handler
    except Exception as e:
        if not cancel_event.is_set():
            await status_message.edit_text(text=f"❌ GoFile upload failed: {escape_markdown_v2(str(e))}", parse_mode='MarkdownV2')
        return None
    finally:
        if not stop_ul_monitor_event.is_set():
            stop_ul_monitor_event.set()
            try: await ul_monitor_task
            except asyncio.CancelledError: pass

# --- Rate Limit Handler ---
def rate_limit_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 1)
            return await func(*args, **kwargs)
        except TelegramError as te: raise te
        except Exception as e: raise e
    return wrapper

# --- Approval Functions ---
def load_approved_groups() -> List[int]:
    if os.path.exists(APPROVED_GROUPS_FILE):
        try:
            with open(APPROVED_GROUPS_FILE, "r") as f: return json.load(f)
        except (json.JSONDecodeError, IOError): pass
    return []
def save_approved_groups(approved_groups: List[int]):
    with open(APPROVED_GROUPS_FILE, "w") as f: json.dump(approved_groups, f, indent=4)
def load_approved_topics() -> Dict[str, List[int]]:
    if os.path.exists(APPROVED_TOPICS_FILE):
        try:
            with open(APPROVED_TOPICS_FILE, "r") as f: return json.load(f)
        except (json.JSONDecodeError, IOError): pass
    return {}
def save_approved_topics(approved_topics: Dict[str, List[int]]):
    with open(APPROVED_TOPICS_FILE, "w") as f: json.dump(approved_topics, f, indent=4)

# --- Telegram Message Handlers ---
@rate_limit_handler
async def _send_message_internal(update_or_message_obj, text, **kwargs):
    target = update_or_message_obj.message if hasattr(update_or_message_obj, 'message') else update_or_message_obj
    return await target.reply_text(text, **kwargs)

# --- Command Handlers ---
async def start(update, context):
    start_message = (
        "Welcome\\! I can download music from Qobuz, Tidal, Deezer, Napster and more\\.\n\n"
        "To download, use the command:\n`/odl <URL>`\n\n"
        "You can specify quality:\n"
        "  `#16bit` for CD\\-quality FLAC \\(16bit/44\\.1kHz\\)\n"
        "  `#lossy` for high\\-quality MP3 \\(320kbps\\)\n"
        "If no tag is used, I will try to download the highest possible quality\\.\n\n"
        "This bot must be run in an approved group or topic\\. "
        "Admins can use `/authg` to approve a group or `/autht` to approve a specific topic\\."
    )
    await _send_message_internal(update, start_message, parse_mode='MarkdownV2')

async def approvegroup(update, context):
    if update.effective_user.id != ADMIN_ID: return
    chat_id = update.effective_chat.id
    approved_groups = load_approved_groups()
    if chat_id in approved_groups:
        await _send_message_internal(update, f"Group `{chat_id}` is already approved\\.", parse_mode='MarkdownV2')
    else:
        approved_groups.append(chat_id)
        save_approved_groups(approved_groups)
        await _send_message_internal(update, f"Group `{chat_id}` has been approved\\.", parse_mode='MarkdownV2')

async def approve_topic(update, context):
    if update.effective_user.id != ADMIN_ID: return
    if not update.message.is_topic_message:
        await _send_message_internal(update, "This command can only be used inside a group topic\\.", parse_mode='MarkdownV2')
        return
    chat_id, thread_id = str(update.effective_chat.id), update.message.message_thread_id
    approved_topics = load_approved_topics()
    if chat_id not in approved_topics: approved_topics[chat_id] = []
    if thread_id in approved_topics[chat_id]:
        await _send_message_internal(update, f"This topic is already approved in group `{chat_id}`\\.", parse_mode='MarkdownV2')
    else:
        approved_topics[chat_id].append(thread_id)
        save_approved_topics(approved_topics)
        await _send_message_internal(update, f"Topic approved successfully in group `{chat_id}`\\.", parse_mode='MarkdownV2')

async def cancel(update, context):
    user_id = update.effective_user.id
    # The download ID is captured by the RegexHandler's replacement
    match = context.matches[0]
    dl_id = match.group(1)

    async with download_tasks_lock:
        if dl_id not in download_registry:
            await _send_message_internal(update, f"Download ID `{escape_markdown_v2(dl_id)}` not found or already completed\\.", parse_mode='MarkdownV2')
            return
        
        registry_entry = download_registry[dl_id]
        if user_id != registry_entry["user_id"] and user_id != ADMIN_ID:
            await _send_message_internal(update, "You are not authorized to cancel this download\\.", parse_mode='MarkdownV2')
            return
        
        registry_entry["cancel_event"].set()
        
        task_status = registry_entry["status"]
        
        if task_status == "processing" and registry_entry["process"] and registry_entry["process"].returncode is None:
            print(f"Cancelling active download {dl_id}. Terminating process PID {registry_entry['process'].pid}.")
            registry_entry["process"].terminate()
        
        elif task_status == "queued":
            print(f"Cancelling queued task {dl_id}.")
            status_message = registry_entry.get("status_message")
            if status_message:
                try:
                    await status_message.edit_text(
                        text=f"✅ Request `{escape_markdown_v2(dl_id)}` was cancelled from the queue\\.",
                        parse_mode='MarkdownV2'
                    )
                except Exception as e:
                    print(f"Could not edit message for cancelled queue item {dl_id}: {e}")
            requesting_user_id = registry_entry["user_id"]
            async with queue_lock:
                if requesting_user_id in user_requests:
                    del user_requests[requesting_user_id]
            del download_registry[dl_id]

    await _send_message_internal(update, f"Cancellation signal sent for download `{escape_markdown_v2(dl_id)}`\\. Please wait\\.", parse_mode='MarkdownV2')

async def dl(update, context):
    user_id, chat_id = update.effective_user.id, update.effective_chat.id
    is_topic, thread_id = update.message.is_topic_message, update.message.message_thread_id
    
    approved_groups = load_approved_groups()
    approved_topics = load_approved_topics()
    is_approved = (update.effective_chat.type == "private" and user_id == ADMIN_ID) or \
                  (chat_id in approved_groups) or \
                  (is_topic and str(chat_id) in approved_topics and thread_id in approved_topics.get(str(chat_id), []))
    if not is_approved:
        await _send_message_internal(update, "This chat or topic is not approved for use\\.", parse_mode='MarkdownV2')
        return

    if not context.args:
        await _send_message_internal(update, "Usage: `/odl <URL> [#lossy|#16bit]`", parse_mode='MarkdownV2')
        return

    url = context.args[0]
    if not any(s in url for s in ["qobuz.com", "deezer.com", "tidal.com", "jiosaavn.com", "beatport.com", "napster.com"]):
        await _send_message_internal(update, f"Invalid URL\\. Use one from supported services\\.", parse_mode='MarkdownV2')
        return

    if user_id != ADMIN_ID and any(p in url for p in ['/artist/', '/discography/', '/interpreter/']):
        await _send_message_internal(update, "Artist/discography links are not supported\\.", parse_mode='MarkdownV2')
        return

    is_playlist = any(s in url for s in ['/playlist/', '/playlists/'])
    quality_tag, is_lossy = "hifi", False
    if len(context.args) > 1:
        if "#lossy" in context.args[1].lower(): quality_tag, is_lossy = "high", True
        elif "#16bit" in context.args[1].lower(): quality_tag, is_lossy = "lossless", False

    async with queue_lock:
        if user_id != ADMIN_ID and user_id in user_requests:
            await _send_message_internal(update, "⏳ You already have a request in progress\\. Please wait\\.", parse_mode='MarkdownV2')
            return

        status_message = await context.bot.send_message(
            chat_id=chat_id, message_thread_id=thread_id,
            text="⏳ Initializing your request\\.\\.\\.",
            reply_to_message_id=update.message.message_id, parse_mode='MarkdownV2'
        )

        dl_id = secrets.token_hex(4)
        queue_size = download_queue.qsize() + 1
        
        async with download_tasks_lock:
            cancel_event = asyncio.Event()
            download_registry[dl_id] = {
                "user_id": user_id,
                "cancel_event": cancel_event,
                "process": None,
                "status": "queued",
                "status_message": status_message,
            }
        
        await download_queue.put((
            user_id, chat_id, thread_id, url, quality_tag,
            is_lossy, update.message.message_id, status_message, is_playlist, dl_id
        ))

        user_requests[user_id] = {"status": "queued", "position": queue_size}

    # --- CHANGE: Modified cancel message format ---
    cancel_info = f"\n\n> ❌ *Cancel:* /cancel\\_{dl_id}"
    # --- CHANGE: Modified status_message to use new cancel_info without extra escaping ---
    await status_message.edit_text(
        text=f"📥 *Queue Position:* {queue_size}\n🆔 *ID:* `{dl_id}`\n🔗 `{escape_markdown_v2(url)}`{cancel_info}",
        parse_mode='MarkdownV2'
    )

# --- Metadata and Zip Functions ---
def update_quality_config(quality):
    config_file = os.path.join("config", "settings.json")
    os.makedirs("config", exist_ok=True)
    try:
        with open(config_file, "r") as f: config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {"global": {"general": {}}}
    config.setdefault("global", {}).setdefault("general", {})
    config["global"]["general"]["download_quality"] = quality
    with open(config_file, "w") as f: json.dump(config, f, indent=4)
    return True

def is_dolby_atmos(file_path: str) -> bool:
    """Checks if an M4A file is likely Dolby Atmos by checking its codec."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".m4a", ".mp4"]:
        return False
    try:
        audio = MP4(file_path)
        # The 'codec' attribute in mutagen's info object for mp4 files
        # will contain 'ec-3' for Enhanced AC-3 tracks, which is used for Atmos.
        if hasattr(audio, 'info') and hasattr(audio.info, 'codec') and 'ec-3' in audio.info.codec.lower():
            print(f"Atmos detected in {file_path} (codec: {audio.info.codec}).")
            return True
    except Exception as e:
        print(f"Could not check for Atmos in {file_path}: {e}")
    return False

def get_audio_metadata(file_path: str) -> Tuple[int, float, float]:
    bit_depth, sample_rate_khz, bitrate_kbps = 16, 44.1, 0.0
    try:
        ext = os.path.splitext(file_path)[1].lower()
        audio = None
        if ext == ".flac": audio = FLAC(file_path)
        elif ext == ".mp3": audio = MP3(file_path)
        elif ext == ".wav": audio = WAVE(file_path)
        elif ext in [".m4a", ".mp4"]: audio = MP4(file_path)
        
        if audio and hasattr(audio, 'info'):
            info = audio.info
            if hasattr(info, 'bits_per_sample') and info.bits_per_sample:
                bit_depth = info.bits_per_sample
            if hasattr(info, 'sample_rate') and info.sample_rate:
                sample_rate_khz = info.sample_rate / 1000.0
            if hasattr(info, 'bitrate') and info.bitrate:
                bitrate_kbps = info.bitrate / 1000.0
    except Exception as e:
        print(f"Error reading tech metadata from {file_path}: {e}")
    return bit_depth, sample_rate_khz, bitrate_kbps

def get_metadata(file_path: str) -> Tuple[str, str, str, str]:
    year, album, album_artist, title = None, "Unknown", "Unknown Artist", "Unknown Title"
    try:
        ext = os.path.splitext(file_path)[1].lower()
        audio = None
        if ext == ".flac": audio = FLAC(file_path)
        elif ext == ".mp3": audio = ID3(file_path)
        elif ext in [".m4a", ".mp4"]: audio = MP4(file_path)

        if not audio:
             if ext == ".mp3": audio = EasyMP3(file_path)
             if not audio: return year, album, album_artist, title
        
        if isinstance(audio, MP4):
            album = audio.get('\xa9alb', [album])[0]
            album_artist = audio.get('aART', [None])[0] or audio.get('\xa9ART', [album_artist])[0]
            title = audio.get('\xa9nam', [title])[0]
            year_val_str = audio.get('\xa9day', [None])[0]
            if year_val_str and (match := re.match(r"(\d{4})", str(year_val_str))):
                year = match.group(1)

        elif isinstance(audio, ID3):
            def get_id3_text(key):
                frame = audio.get(key)
                return str(frame.text[0]) if frame and frame.text else None
            album = get_id3_text('TALB') or album
            album_artist = get_id3_text('TPE2') or get_id3_text('TPE1') or album_artist
            title = get_id3_text('TIT2') or title
            year_val_str = get_id3_text('TDRC') or get_id3_text('TDOR') or get_id3_text('TYER')
            if year_val_str and (match := re.match(r"(\d{4})", str(year_val_str))):
                year = match.group(1)

        elif isinstance(audio, FLAC):
            def get_vorbis(key): return audio.get(key, [None])[0]
            album = get_vorbis('album') or album
            album_artist = get_vorbis('albumartist') or get_vorbis('artist') or album_artist
            title = get_vorbis('title') or title
            year_val_str = get_vorbis('date') or get_vorbis('originaldate')
            if year_val_str and (match := re.match(r"(\d{4})", str(year_val_str))):
                year = match.group(1)

    except Exception as e:
        print(f"Error reading metadata for {file_path}: {str(e)}")
    
    return str(year) if year else None, str(album), str(album_artist), str(title)

def create_zip(files_to_zip: List[str], zip_name_unsafe: str) -> str:
    if not files_to_zip:
        return None

    zip_name = sanitize_filename(zip_name_unsafe)
    zip_path = os.path.join(TEMP_FOLDER, zip_name)
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            arcname = os.path.relpath(file_path, DOWNLOAD_FOLDER)
            zipf.write(file_path, arcname)
            
    return zip_path

def sanitize_filename(name):
    if not name: return "Unknown_Filename"
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)
    name = re.sub(r'[_\s]+', ' ', name).strip()
    return name[:180] if name else "Sanitized_Filename"

# --- Single Instance Check ---
def check_single_instance():
    lock_file = "bot.lock"
    if sys.platform == "win32":
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return os.fdopen(fd, "w")
        except FileExistsError: sys.exit("Lock file exists. Another instance running?")
    else: # POSIX
        import fcntl
        fp = open(lock_file, "w")
        try:
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fp
        except BlockingIOError:
            fp.close()
            sys.exit("Failed lock via fcntl. Another instance running?")

# --- Main Function ---
def main():
    lock_fp = check_single_instance()
    print("Instance lock acquired.")
    try:
        for folder in [TEMP_FOLDER, DOWNLOAD_FOLDER, "config"]: os.makedirs(folder, exist_ok=True)
        app = Application.builder().token(BOT_TOKEN).build()
        app.job_queue.run_once(lambda _: asyncio.create_task(queue_processor(app)), 1)
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("odl", dl))
        # --- CHANGE: Updated regex to handle commands with @BotUsername ---
        app.add_handler(MessageHandler(filters.Regex(r'^/cancel_([a-f0-9]{8})(?:@\w+)?$'), cancel))
        app.add_handler(CommandHandler("authg", approvegroup))
        app.add_handler(CommandHandler("autht", approve_topic))
        print("Bot polling...")
        app.run_polling()
    except SystemExit as e: print(f"Exiting: {e}")
    except Exception as e: traceback.print_exc()
    finally:
        if lock_fp:
            print("Releasing instance lock.")
            lock_fp.close()
            if os.path.exists("bot.lock"): os.remove("bot.lock")
        print("Bot shutdown.")

if __name__ == "__main__":
    if not os.path.exists("orpheus.py"):
        print("CRITICAL: orpheus.py not found. Bot requires it to function.")
        sys.exit(1)
    main()
