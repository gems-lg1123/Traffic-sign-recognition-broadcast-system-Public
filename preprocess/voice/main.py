# 导入pyttsx3库
import pyttsx3


class Voice():
      def __init__(self):
            self.engine = pyttsx3.init()  # 创建engine并初始化
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[3].id) # 英语（英国）的女性语音
            self.engine.setProperty('rate', 140)
            self.engine.setProperty('volume', 1.0)  # 在0到1之间重设音量

      def synthesis(self, text, filename):
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()

      def play(self, filename):
            self.engine.say(filename)
            self.engine.runAndWait()
            self.engine.stop()


if __name__ == "__main__":
    speech = Voice()
    names = [ 'Speed limit (20km/h)', 'Speed limit (30km/h)','Speed limit (50km/h)', 'Speed limit (60km/h)',
              'Speed limit (70km/h)','Speed limit (80km/h)', 'End of speed limit (80km/h)','Speed limit (100km/h)',
              'Speed limit (120km/h)','No passing',  'No passing vehicle over 3.5 tons','Right-of-way at intersection',
              'Priority road',  'Yield', 'Stop','No vehicles', 'Vehicle > 3.5 tons prohibited', 'No entry',
              'General caution', 'Dangerous curve left','Dangerous curve right', 'Double curve', 'Bumpy road',
              'Slippery road', 'Road narrows on the right', 'Road work',  'Traffic signals', 'Pedestrians',
              'Children crossing','Bicycles crossing','Beware of ice/snow', 'Wild animals crossing', 'passing limit',
              'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
              'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing','End no passing vehicle > 3.5 tons']

    for i in range(0,43):
           address = str(i) + '.mp3'
           speech.synthesis(names[i], address) #此句是在当前文件目录下生成mp3文件
    det=[1,2,3,1]
    cname = int(det[:, -1])
    name = str(cname) + '.mp3'
    print(name)
speech.play(names[1]) #此句是直接合成语音并播放
