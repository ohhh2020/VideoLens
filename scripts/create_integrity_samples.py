"""生成三类10分钟、带确定事实标注的受控测试视频；不是自然视频质量证明。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from PIL import Image, ImageDraw, ImageFont
    from backend.media.ffmpeg_utils import resolve_binary
    parser = argparse.ArgumentParser(description='Windows语音合成与FFmpeg生成隔离测试样本')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries, samples = [], []
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 48)
    small = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 32)
    for category, count in [('sparse', 3), ('medium', 12), ('dense', 60)]:
        directory = root / category
        directory.mkdir(exist_ok=True)
        facts = []
        for index in range(count):
            value, limit = 100000 + (index+1)*137, 200000 + (index+1)*173
            slide = directory / f'{index:03d}.png'
            image = Image.new('RGB', (1920,1080), '#f4f5f6')
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((65,65,1855,1015),30,fill='white',outline='#d0d3d4',width=2)
            draw.text((125,115), f'VideoLens controlled sample / {category}', font=font, fill='#1c2922')
            draw.text((125,230), f'ITEM_{index+1:03d}: explicit numeric facts', font=font, fill='#171b19')
            draw.text((125,350), f'value_{index+1:03d} = {value}', font=font, fill='#171b19')
            draw.text((125,450), f'limit_{index+1:03d} = {limit}', font=font, fill='#171b19')
            if category == 'dense':
                for line, text in enumerate(['def evaluate(value, limit):', '    if value < limit:', '        return "accepted"', '    return "review"']):
                    draw.text((125,580+line*58), text, font=small, fill='#222c26')
                draw.rectangle((1100,330,1750,760), outline='#414b44', width=3)
                for line, text in enumerate(['column | observed', '-----------------', f'value  | {value}', f'limit  | {limit}', 'status | accepted']):
                    draw.text((1140,370+line*66),text,font=small,fill='#171b19')
            draw.text((125,900), 'Test material: transcribe values exactly; do not infer missing data.', font=small, fill='#444b46')
            image.save(slide)
            wav = directory / f'{index:03d}.wav'
            entries.append({'wav': str(wav), 'text': f'Item {index+1}. The value is {value}. The limit is {limit}.'})
            facts.extend([{'item': index+1, 'key': f'value_{index+1:03d}', 'value': str(value), 'start_ms': index*600000//count},
                          {'item': index+1, 'key': f'limit_{index+1:03d}', 'value': str(limit), 'start_ms': index*600000//count}])
        samples.append({'category': category, 'count': count, 'path': str(root/(category+'.mp4')), 'facts': facts,
                        'note': '受控合成幻灯片与Windows TTS，1920x1080；不可替代自然视频人工验收。'})
    (root/'speech.json').write_text(json.dumps(entries),encoding='utf-8')
    script = root/'synthesize.ps1'
    script.write_text('''param([string]$Manifest)
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.SelectVoice("Microsoft Zira Desktop")
$speaker.Rate = 2
try { foreach ($entry in (Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json)) {
  if (!(Test-Path -LiteralPath $entry.wav)) { $speaker.SetOutputToWaveFile($entry.wav); $speaker.Speak($entry.text); $speaker.SetOutputToNull() }
} } finally { $speaker.Dispose() }
''',encoding='utf-8-sig')
    powershell = 'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    subprocess.run([powershell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),'-Manifest',str(root/'speech.json')],check=True)
    for sample in samples:
        parts = []
        directory = root/sample['category']
        for index in range(sample['count']):
            part = directory/f'{index:03d}.mp4'
            if not part.is_file():
                subprocess.run([resolve_binary('ffmpeg'),'-y','-nostdin','-loglevel','error','-loop','1','-framerate','2',
                    '-i',str(directory/f'{index:03d}.png'),'-i',str(directory/f'{index:03d}.wav'),
                    '-vf','format=yuv420p','-af','apad','-t',str(600/sample['count']),'-c:v','libx264','-preset','veryfast','-crf','18',
                    '-threads','2','-c:a','aac','-ar','16000',str(part)],check=True)
            parts.append("file '"+part.as_posix().replace("'","'\\''")+"'")
        playlist=directory/'concat.txt';playlist.write_text('\n'.join(parts),encoding='utf-8')
        subprocess.run([resolve_binary('ffmpeg'),'-y','-nostdin','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),
                         '-c','copy',sample['path']],check=True)
        print(sample['category'],sample['path'],flush=True)
    (root/'manifest.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
