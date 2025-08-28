import gradio as gr
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cv2
import tempfile
import os
from pathlib import Path

def create_spectrogram_video(audio_file, video_duration=10, fps=30, colormap='viridis'):
    """
    Create a scrolling spectrogram video from an audio file.
    
    Args:
        audio_file: Path to the input WAV file
        video_duration: Duration of the output video in seconds
        fps: Frames per second for the output video
        colormap: Colormap for the spectrogram
    
    Returns:
        Path to the generated video file
    """
    
    # Load audio file
    y, sr = librosa.load(audio_file)
    
    # Compute spectrogram
    hop_length = 512
    n_fft = 2048
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # Get time and frequency axes
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Calculate video parameters
    total_frames = int(video_duration * fps)
    time_per_frame = len(times) / total_frames
    
    # Set up the figure with no margins
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.style.use('dark_background')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    
    # Create temporary video file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, 'spectrogram_video.mp4')
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1200, 800))
    
    # Window size for scrolling effect
    window_size = min(200, S_db.shape[1] // 4)  # Show 1/4 of the spectrogram at a time
    
    # Calculate total scrolling range (start from off-screen right, end off-screen left)
    total_scroll_frames = S_db.shape[1] + window_size
    scroll_per_frame = total_scroll_frames / total_frames
    
    for frame_idx in range(total_frames):
        ax.clear()
        
        # Calculate scrolling position (starts at -window_size, ends at S_db.shape[1])
        scroll_position = int(frame_idx * scroll_per_frame - window_size)
        start_frame = max(0, scroll_position)
        end_frame = min(S_db.shape[1], scroll_position + window_size)
        
        # Handle padding for off-screen portions
        if scroll_position < 0:
            # Pad with zeros on the left (off-screen right at start)
            pad_left = abs(scroll_position)
            current_spec = S_db[:, start_frame:end_frame]
            current_spec = np.pad(current_spec, ((0, 0), (pad_left, 0)), mode='constant', constant_values=S_db.min())
            
            # Create corresponding time array with padding
            visible_times = times[start_frame:end_frame]
            if len(visible_times) > 0:
                time_step = visible_times[1] - visible_times[0] if len(visible_times) > 1 else 0.1
                pad_times = np.arange(visible_times[0] - pad_left * time_step, visible_times[0], time_step)
                current_times = np.concatenate([pad_times, visible_times])
            else:
                current_times = np.linspace(0, window_size * 0.01, window_size)
                
        elif scroll_position + window_size > S_db.shape[1]:
            # Pad with zeros on the right (off-screen left at end)
            pad_right = scroll_position + window_size - S_db.shape[1]
            current_spec = S_db[:, start_frame:end_frame]
            current_spec = np.pad(current_spec, ((0, 0), (0, pad_right)), mode='constant', constant_values=S_db.min())
            
            # Create corresponding time array with padding
            visible_times = times[start_frame:end_frame]
            if len(visible_times) > 0:
                time_step = visible_times[1] - visible_times[0] if len(visible_times) > 1 else 0.1
                pad_times = np.arange(visible_times[-1] + time_step, visible_times[-1] + (pad_right + 1) * time_step, time_step)
                current_times = np.concatenate([visible_times, pad_times])
            else:
                current_times = np.linspace(0, window_size * 0.01, window_size)
        else:
            # Normal case - fully within spectrogram
            current_spec = S_db[:, start_frame:end_frame]
            current_times = times[start_frame:end_frame]
        
        # Ensure we have the right window size
        if current_spec.shape[1] != window_size:
            if current_spec.shape[1] < window_size:
                pad_needed = window_size - current_spec.shape[1]
                current_spec = np.pad(current_spec, ((0, 0), (0, pad_needed)), mode='constant', constant_values=S_db.min())
                if len(current_times) > 0:
                    time_step = current_times[1] - current_times[0] if len(current_times) > 1 else 0.01
                    pad_times = np.arange(current_times[-1] + time_step, current_times[-1] + (pad_needed + 1) * time_step, time_step)
                    current_times = np.concatenate([current_times, pad_times])
                else:
                    current_times = np.linspace(0, window_size * 0.01, window_size)
        
        # Plot spectrogram
        im = ax.imshow(current_spec, 
                      aspect='auto', 
                      origin='lower',
                      extent=[current_times[0], current_times[-1], freqs[0], freqs[-1]],
                      cmap=colormap,
                      vmin=S_db.min(),
                      vmax=S_db.max())
        
        # Remove all axes elements for clean full-screen display
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(current_times[0], current_times[-1])
        ax.set_ylim(freqs[0], freqs[-1])
        
        # Remove all spines/borders
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Convert matplotlib figure to OpenCV format
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]  # Remove alpha channel
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (1200, 800))
        
        # Write frame to video
        video_writer.write(frame_resized)
    
    # Clean up
    video_writer.release()
    plt.close(fig)
    
    return video_path

def get_audio_duration(audio_file):
    """
    Get the duration of an audio file in seconds.
    """
    if audio_file is None:
        return None
    
    try:
        y, sr = librosa.load(audio_file)
        duration = len(y) / sr
        return duration
    except Exception:
        return None

def process_audio_file(audio_file, fps, colormap):
    """
    Process the uploaded audio file and create spectrogram video.
    """
    try:
        if audio_file is None:
            return None, "Please upload a WAV file."
        
        # Get audio duration to use as video duration
        audio_duration = get_audio_duration(audio_file)
        if audio_duration is None:
            return None, "Could not determine audio duration."
        
        # Create the video with audio duration
        video_path = create_spectrogram_video(
            audio_file, 
            video_duration=audio_duration, 
            fps=fps, 
            colormap=colormap
        )
        
        return video_path, "Video generated successfully!"
    
    except Exception as e:
        return None, f"Error processing audio file: {str(e)}"


def create_gradio_app():
    with gr.Blocks(title="Spectrogram Video Generator") as app:
        gr.Markdown("# Spectrogram Video Generator")
        gr.Markdown("Upload a WAV file to create a scrolling spectrogram video visualization.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                audio_input = gr.Audio(
                    label="Upload WAV File",
                    type="filepath",
                    format="wav"
                )
                
                fps_slider = gr.Slider(
                    minimum=15,
                    maximum=60,
                    value=30,
                    step=1,
                    label="Frames Per Second"
                )
                
                colormap_dropdown = gr.Dropdown(
                    choices=["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "spring", "summer", "autumn", "winter"],
                    value="inferno",
                    label="Colormap"
                )
                
                generate_btn = gr.Button("Generate Video", variant="primary")
            
            with gr.Column(scale=2):
                # Output components
                video_output = gr.Video(label="Generated Spectrogram Video")
                status_output = gr.Textbox(label="Status", interactive=False)
        
        
        # Set up the event handler
        generate_btn.click(
            fn=process_audio_file,
            inputs=[audio_input, fps_slider, colormap_dropdown],
            outputs=[video_output, status_output]
        )
        
        gr.Markdown("""
        ## How it works:
        1. Upload a WAV audio file
        2. Adjust frame rate and colormap
        3. Click "Generate Video" to create a scrolling spectrogram visualization
        4. The video will show a time-scrolling view of the audio's frequency content
        
        **Note:** Video duration will match the audio file duration. Processing may take a few minutes depending on file size.
        """)
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)