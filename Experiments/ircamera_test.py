from flask import Flask, Response, render_template_string
import cv2
import RPi.GPIO as GPIO
import threading
import time

app = Flask(__name__)

# === [하드웨어 설정] ===
IR_PIN = 17  # 물리 11번 핀
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.OUT)

# 초기 상태
current_mode = False
GPIO.output(IR_PIN, current_mode)

# === [카메라 설정] ===
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # 640x480 해상도 (밸런스)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        self.grabbed, self.frame = self.video.read()
        self.stopped = False
        
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.stopped = True
        self.video.release()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.video.read()

    def get_frame(self):
        if not self.grabbed:
            return None
        
        image = self.frame
        
        # [수정 완료] 이제 화면 상태와 글자가 일치할 겁니다!
        if current_mode:
            # GPIO High일 때 -> 필터 닫힘(추정) -> 레이저 작게 보임
            status = "Day Mode (Filter ON)"
            color = (0, 255, 0) # 초록색 글씨
        else:
            # GPIO Low일 때 -> 필터 열림(추정) -> 레이저 번져 보임 (Blooming)
            status = "Night Mode (NoIR)"
            color = (255, 0, 255) # 보라색 글씨
            
        cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        ret, jpeg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpeg.tobytes()

video_camera = None

def start_camera():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()

@app.route('/')
def index():
    # 웹페이지 안내 문구도 수정
    if current_mode:
        state_text = "현재: 주간 모드 (필터 ON - 레이저 작음)"
        btn_color = "#28a745" # 초록 버튼
    else:
        state_text = "현재: 야간 모드 (필터 OFF - 레이저 번짐!)"
        btn_color = "#6f42c1" # 보라 버튼

    return render_template_string('''
        <body style="background:#222; color:white; text-align:center; font-family:sans-serif;">
            <h2>✅ Final Corrected View</h2>
            <img src="/video_feed" style="border:2px solid #fff; width:640px; height:480px;"><br><br>
            <form action="/toggle" method="post">
                <button style="padding:15px 40px; font-size:18px; background:{{ btn }}; color:white; border:none; border-radius:5px; cursor:pointer;">
                    Switch Mode (딸깍!)
                </button>
            </form>
            <h3>{{ state }}</h3>
            <p style="color:#aaa; font-size:0.9em;">* 이제 레이저가 번져 보일 때 'Night Mode'라고 뜰 겁니다.</p>
        </body>
    ''', state=state_text, btn=btn_color)

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    start_camera()
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle', methods=['POST'])
def toggle():
    global current_mode
    current_mode = not current_mode
    GPIO.output(IR_PIN, current_mode)
    return index()

if __name__ == '__main__':
    start_camera()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)