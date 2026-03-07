from moviepy.editor import VideoFileClip
import os

videos = [
    'report/videos/diffusion/diffusion_seed100000.mp4',
    'report/videos/diffusion/diffusion_seed100001.mp4',
    'report/videos/regression/regression_seed100000.mp4',
    'report/videos/regression/regression_seed100001.mp4'
]

for video in videos:
    if os.path.exists(video):
        gif_path = video.replace('.mp4', '.gif')
        clip = VideoFileClip(video)
        clip.write_gif(gif_path, fps=10)
        print(f"Converted {video} to {gif_path}")
