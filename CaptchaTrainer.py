import requests
import os
import PIL.Image
import PIL.ImageTk
import shutil
from tkinter import *
from tkinter import messagebox
from captcha import *
class CaptchaTrainer(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.decoder = CaptchaDecoder()
		self.grid()
		self.createWidgets()
		self.getAndCheck()
	
	def createWidgets(self):
		self.img = PIL.ImageTk.PhotoImage(PIL.Image.open("pic.jpg"))
		self.imageLabel = Label(self,image=self.img)
		self.imageLabel.grid(row=0, column=0)
		
		self.displayText = Label(self)
		self.displayText["text"] = "something happened"
		self.displayText.grid(row=1, column=0, columnspan=4)
		
		self.imageField = Entry(self)
		self.imageField["width"] = 10
		self.imageField.grid(row=2, column=0, columnspan=1)
		
		submit = Button(self)
		submit["text"] = "Ok"
		submit["command"] = self.submitResult
		submit.grid(row=3,column=0)
		
	def submitResult(self):
		tmplist = os.listdir('tmp')
		input = self.imageField.get()
		if len(tmplist) == len(input):
			for i in range(len(input)):
				c = input[i]
				filelist = os.listdir('templates/'+c)
				if filelist:
					last = filelist[len(filelist)-1].split('.')
					num = int(last[0])+1
				else:
					num = 0
				if num < 10:
					num = '000'+str(num)
				elif num < 100:
					num = '00' + str(num)
				elif num < 1000:
					num = '0' + str(num)
				filename = 'templates/'+c+'/'+num+'.png'
				print(filename)
				shutil.copy('tmp/'+tmplist[i],filename)
				
		#else:
			#messagebox.showinfo("Error","length not match")
			
		for f in tmplist:
			os.remove('tmp/'+f)
		self.getAndCheck()
	
	def getAndCheck(self):
		image_url = "http://railway.hinet.net/ImageOut.jsp"
		r = requests.get(image_url)
		with open("img.jpg","wb") as f:
			f.write(r.content)
		str = self.decoder.identify("img.jpg")
		self.displayText['text'] = str
		self.img = PIL.ImageTk.PhotoImage(PIL.Image.open("img.jpg"))
		self.imageLabel['image'] = self.img
	
if __name__ == '__main__':
	root = Tk()
	root.title('TicketBooker')
	app = CaptchaTrainer(master=root)
	app.mainloop()
