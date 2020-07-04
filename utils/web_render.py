import flask
import numpy as np
import threading
from flask import Response
from flask import Flask
from flask import render_template
import logging
import time
import cv2

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2plot


class WebRenderer(object):

    def __init__(self, 
                 port,
                 batch_size,
                 sample_nums, 
                 update_every_batches, 
                 total_epoches=90, 
                 mode='auto', 
                 blank_size=70, 
                 epoch_pixel=20, 
                 max_vis_loss=6.4,
                 canvas_h=500,
                 x_ruler=4,
                 y_ruler=2):

        """

        Inputs:
            port : decide which port you want to use.
            batch_size : training batch_size
            sample_nums : dataset size (how many training sample?)
            update_every_batches : every batches update loss or accuracy when training once.
                ==> recommand is sample_nums/5 ~ sample_nums/10
            total_epoches : total training epoches.
            mode = 'auto'
            blank_size : the border size of your canvas.
            epoch_pixel : how many pixel in one epoches (this decide canvas's width)
            max_vis_loss : the max loss value you want to show.
            canvas_h : canvas height. (fixed)
            x_ruler : how many lines you want to show in one epoch.
            y_ruler : how many lines you want to show in.

        Paramaters
            app : flask application (to render web page)
            port : 127.0.0.1:port
            out_frame : array to render
            lock : threading prevent
            isbreak : can break loop from main thread.

        """


        self.canvas = cv2plot.Canvas(batch_size,
                                     sample_nums, 
                                     update_every_batches, 
                                     total_epoches=90, 
                                     mode='auto', 
                                     blank_size=70, 
                                     epoch_pixel=20, 
                                     max_vis_loss=6.4,
                                     canvas_h=500,
                                     x_ruler=4,
                                     y_ruler=2)
        
        self.app = Flask(__name__)
        self.port = port

        self.out_frame = self.canvas.background.copy()

        self.lock = threading.Lock()
        self.isbreak = False

        self.create_html()
        self.web_page_setting()

    def web_page_setting(self):
        """
        set web url.
        """
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/frame_feed', 'frame_feed', self.frame_feed)
        self._disable_log() # comment this line will show all your web client.

    def _disable_log(self):
        log = logging.getLogger('werkzeug')
        log.disabled = True
        self.app.logger.disabled = True

    def start(self):
        """
            call this function to start web server.
        """
        self.app.run(host="127.0.0.1", port=self.port, debug=True,
            threaded=True, use_reloader=False)

    def updating(self, acc=None, loss=None, show_this=False, mode='train'):

        with self.lock:
            if len(acc) == 1:
                self.canvas.plot_list(loss, [show_this], dot_color=(200,0,0))
                self.canvas.plot_list(acc, [show_this], dot_color=(0,0,200))
            elif len(acc) == 2:
                self.canvas.plot_list(loss, [False, show_this], dot_color=(200,0,0))
                self.canvas.plot_list(acc, [False, show_this], dot_color=(0,0,200))

            self.out_frame = self.canvas.background.copy()

    def generating(self):
        """
        yield frames to assign url.
        """
        while True:
            if self.isbreak:
                break
            try:
                with self.lock:
                    if self.out_frame is not None:
                        (flag, encodedImage) = cv2.imencode(".jpg", self.out_frame)
                        if flag:
                            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                                bytearray(encodedImage) + b'\r\n')
                time.sleep(1)
            except:
                break

    def index(self):
        """ render to url: '/' """
        return render_template("index.html")

    def frame_feed(self):
        """ response to url: '/frame_feed' """
        return Response(self.generating(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def create_html(self):
        """
        this function would auto write a main html of this web application.
        """
        abs_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(abs_path+"/templates/"):
            os.mkdir(abs_path+"/templates/")
        if not os.path.isfile("/templates/index.html"):
            with open(abs_path+"/templates/index.html", "w") as fhtml:
                fhtml.write('<html>\n')
                fhtml.write('  <head>')
                fhtml.write('    <title>training acc and loss</title>\n')
                fhtml.write('  </head>\n')
                fhtml.write('  <body>\n')
                fhtml.write('    <img src="/frame_feed">\n')
                fhtml.write('  </body>\n')
                fhtml.write('</html>\n')


