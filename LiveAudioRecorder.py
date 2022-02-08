import pandas as pd
import numpy as np
import librosa.display
import pyaudio
import audioop
import wave
import keyboard
import onnxruntime


CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output.wav"
SILENT_CHUNKS = 2 * RATE / CHUNK # 2초 이상 silent
THRESHOLD = 100

gender = ['man', 'woman']
# device_idx = int(input("마이크 번호: "))

class LiveAudio(object):
    def __init__(self, device_index=0):
        super(LiveAudio, self).__init__()
        self.py_audio = pyaudio.PyAudio()

        self.stream = self.py_audio.open(format=FORMAT,
                                         channels=CHANNELS,
                                         rate=RATE,
                                         input=True,
                                         frames_per_buffer=CHUNK,
                                         input_device_index=device_index)
        
        self.audio_analysis_size = int(RATE / CHUNK * RECORD_SECONDS)
        self.model = onnxruntime.InferenceSession('model/22050_20_gender_model.onnx', None)
        print("enter 'ESC' to exit")

    def is_silence(self, input_data):
        rms = audioop.rms(input_data, 2)
        return rms < THRESHOLD

    def run(self):
        frames = []

        silence_cnt = 0
        record_start = False

        while(True):
            # datas = np.fromstring(stream.read(CHUNK),dtype=np.int16)
            # print("np: ", int(np.average(np.abs(datas))))
            data = self.stream.read(CHUNK) # len:4096
            silent = self.is_silence(data)

            if record_start:
                if silent:
                    silence_cnt += 1
                    if silence_cnt > SILENT_CHUNKS:
                        frames.clear()
                        print('not analysis..')
                    continue
                else:
                    silence_cnt = 0

            elif not silent:
                record_start = True
                print('* recording')
            
            frames.append(data)
            
            if len(frames) == self.audio_analysis_size:
                print("* done recording")
                self.save_to_wave(frames)
                self.predict_gender()
                frames.clear()
                record_start = False

            if keyboard.is_pressed("ESC"):
                print("exit...")
                break

    def predict_gender(self):
        data, sr = librosa.load(WAVE_OUTPUT_FILENAME, sr=RATE, duration=2.5)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20), axis=0)
        mfcc = pd.DataFrame(data=mfccs)
        mfcc = mfcc.stack().to_frame().T

        input_data = np.expand_dims(np.array(mfcc, dtype=np.float32), axis=2)
        input_data = input_data if isinstance(input_data, list) else [input_data]
        feed = dict([(input.name, input_data[n]) for n, input in enumerate(self.model.get_inputs())])

        y_pred = self.model.run(None, feed)[0].squeeze()
        i_pred = int(np.argmax(y_pred))
        print(f"predict gender: {gender[i_pred]}")


    def save_to_wave(self, data):
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.py_audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(data))
        wf.close()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.py_audio.terminate()


if __name__ == '__main__':
    audiocap = LiveAudio()
    audiocap.run()
    audiocap.stop()