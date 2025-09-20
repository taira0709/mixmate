import sys, io, json
import numpy as np
import soundfile as sf
sys.path.insert(0, '.')

from app import app

# テスト音源作成（3秒）
sr = 44100
t = np.arange(sr * 3, dtype=np.float32) / sr
audio = 0.2 * np.sin(2 * np.pi * 440.0 * t)
buf = io.BytesIO()
sf.write(buf, audio, sr, format='WAV')
wav_bytes = buf.getvalue()

print('Testing audio processing...')

with app.test_client() as client:
    # 1. ファイルアップロード
    rv_upload = client.post('/upload', data={
        'file': (io.BytesIO(wav_bytes), 'test.wav')
    }, content_type='multipart/form-data')
    
    if rv_upload.status_code == 200:
        fid = rv_upload.get_json()['file_id']
        print('✓ Upload successful, file_id:', fid)
        
        # 2. エフェクトOFFでプレビュー
        rv_off = client.post('/process', data={
            'file_id': fid,
            'gender': 'male',
            'style': 'pop',
            'fx_on': json.dumps({'eq': False, 'comp1': False, 'comp2': False}),
            'mode': 'preview'
        })
        
        # 3. エフェクトONでプレビュー  
        rv_on = client.post('/process', data={
            'file_id': fid,
            'gender': 'male', 
            'style': 'pop',
            'fx_on': json.dumps({'eq': True, 'comp1': True, 'comp2': True}),
            'mode': 'preview'
        })
        
        if rv_off.status_code == 200 and rv_on.status_code == 200:
            tp_off = float(rv_off.headers.get('X-TP-DB', '0'))
            tp_on = float(rv_on.headers.get('X-TP-DB', '0'))
            gr1_off = float(rv_off.headers.get('X-GR1-PEAK-DB', '0'))
            gr1_on = float(rv_on.headers.get('X-GR1-PEAK-DB', '0'))
            
            print(f'✓ Effects OFF: TP={tp_off:.1f}dB, GR1={gr1_off:.1f}dB')
            print(f'✓ Effects ON:  TP={tp_on:.1f}dB, GR1={gr1_on:.1f}dB')
            
            if abs(tp_on - tp_off) > 0.5 or gr1_on > 1.0:
                print('✓ Effects are working - audio processing is functional')
            else:
                print('⚠ Effects may not be working properly')
        else:
            print('✗ Audio processing failed')
            print('OFF status:', rv_off.status_code)
            print('ON status:', rv_on.status_code)
    else:
        print('✗ Upload failed:', rv_upload.status_code)
