# Spectrogram Video Generator

A Gradio web application that creates scrolling spectrogram videos from audio files. Upload any audio file and watch as its frequency content is visualized in a beautiful, time-scrolling spectrogram animation.

Tested with Python 3.12.9

## Features

- **Multiple Audio Format Support**: Works with MP3, WAV, FLAC, M4A, OGG, and other librosa-compatible formats
- **Customizable Visualizations**: Choose from 11 different colormaps (viridis, plasma, inferno, magma, etc.)
- **Adjustable Frame Rate**: Set FPS from 15 to 60 for smooth or detailed animations
- **Audio Integration**: Option to include original audio track in the generated video
- **Automatic Duration Matching**: Video duration automatically matches the input audio length
- **Full-Screen Visualization**: Clean, borderless spectrogram display optimized for video

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bug-free-carnival.git
   cd bug-free-carnival
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install ffmpeg for audio integration:
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`)

3. Upload an audio file using the file upload interface

4. Adjust settings:
   - **Frames Per Second**: Higher values create smoother animations (15-60 FPS)
   - **Colormap**: Choose your preferred color scheme for the spectrogram
   - **Include Audio**: Toggle whether to include the original audio in the output video

5. Click "Generate Video" and wait for processing to complete

## How It Works

The application uses the following process:

1. **Audio Analysis**: Loads the audio file and computes its Short-Time Fourier Transform (STFT)
2. **Spectrogram Generation**: Converts the STFT to a dB-scaled spectrogram showing frequency content over time
3. **Video Creation**: Creates a scrolling animation that moves through the spectrogram from right to left
4. **Audio Integration**: Optionally combines the visualization with the original audio track using ffmpeg

## Technical Details

- **Window Size**: Shows 1/4 of the total spectrogram at any given time
- **Scrolling Animation**: Smooth transition from off-screen right to off-screen left
- **Resolution**: Output videos are rendered at 1200x800 pixels
- **Processing**: Uses matplotlib for visualization and OpenCV for video encoding

## Dependencies

- **gradio**: Web interface framework
- **librosa**: Audio processing and analysis
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computing
- **opencv-python**: Video processing
- **scipy**: Scientific computing utilities

## Notes

- Processing time depends on audio file length and chosen frame rate
- Videos without audio are generated even if ffmpeg is not installed
- The application automatically handles various audio formats through librosa
- Generated videos are temporarily stored and cleaned up after download
