import cv2
import numpy as np
import time


class Canvas(object):

    def __init__(self, 
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
            about canvas size setting:
                blank_size : border size
                epoch_pixel : number of pixels in one epoch (x-axis).
                max_vis_loss : is upper bound of losses in canvas.
                w, h, total_w, total_h
        """
        self.batch_size = batch_size
        self.sample_nums = sample_nums
        self.update_every_batches = update_every_batches
        self.total_epoches = total_epoches
        self.mode = mode

        self.blank_size = blank_size
        self.epoch_pixel = epoch_pixel
        self.max_vis_loss = max_vis_loss
        self.w, self.h = int(total_epoches*self.epoch_pixel), canvas_h
        self.total_w, self.total_h = int(self.w + self.blank_size), int(self.h + self.blank_size)
        self.batch_step = self.get_batch_step()

        self.draw_line_in_epoch = x_ruler
        self.draw_line_in_loss = y_ruler

        self.background = self.plot_bg(batch_size, sample_nums, update_every_batches, total_epoches, mode)


    def get_batch_step(self):
        batches_in_epoch = self.sample_nums/self.batch_size # how many batches ~= epoch ?
        batches_in_epoch = batches_in_epoch/self.update_every_batches # how often your model update loss once (under training)
        batch_step = self.epoch_pixel/batches_in_epoch
        return batch_step

    def data2plotpos(self, x, y, xtype='batch', ytype='loss'):
        if xtype in ['b', 'batch']:
            x = int((x*self.batch_step) + self.blank_size/2)
        elif xtype in ['e', 'epoch']:
            x = int(((x+1) * self.epoch_pixel) + self.blank_size/2)

        if ytype in ['l', 'loss']:
            y = int(self.total_h - y*self.h/self.max_vis_loss - self.blank_size/2)
        elif ytype in ['a', 'acc', 'accuracy']:
            y = int(self.total_h - y*self.h/100 - self.blank_size/2)

        x = min(max(0,x), self.total_w) # fits in background range
        y = min(max(0,y), self.total_h)
        return x, y

    def plot_bg(self, batch_size, sample_nums, update_every_batches, total_epoches=90, mode='auto'):
        """
          this function will return a uint8 array with size (max_epoches*100)x500x3(WxHxC),
          auto fit epoch<---->batch size and loss<--->accuracy size.
        """
        assert mode == 'auto'

        bg = np.ones([self.total_h, self.total_w, 3], dtype=np.uint8)*255 # white background
        self.draw_bg_x(bg, "first") # draw line and put text for vertical line
        self.draw_bg_y(bg, "first")
        self.draw_bg_x(bg, "second")
        self.draw_bg_y(bg, "second")

        return bg

    def _txt_d(self, msg):
        d = 0
        if len(msg) == 1:
            d = 3
        elif len(msg) == 2:
            d = 7
        elif len(msg) == 3:
            d = 10
        elif len(msg) == 4:
            d = 13
        return d

    def draw_bg_x(self, bg, line_type):

        x_pos = self.blank_size/2
        min_y, max_y = int(self.total_h - self.blank_size/2), int(self.blank_size/2)
        
        if line_type == "first":
            # first draw light line
            draw_line_in_epoch = self.draw_line_in_epoch
            while x_pos<((self.total_w-self.blank_size/2)+1e-3):
                cv2.line(bg, (int(x_pos), min_y), (int(x_pos), max_y), (180,180,180), 1)
                x_pos += self.epoch_pixel/draw_line_in_epoch

        if line_type == "second":
            # draw black line (vertical)
            epoch_count = 0
            while x_pos<((self.total_w-self.blank_size/2)+1e-3):
                cv2.line(bg, (int(x_pos), min_y), (int(x_pos), max_y), (0,0,0), 1)
                msg = str(epoch_count)
                d = self._txt_d(msg)
                cv2.putText(bg, msg, (int(x_pos)-d, min_y+15), 1, 0.8, (0,0,0), 1)
                x_pos += self.epoch_pixel
                epoch_count += 1

    def draw_bg_y(self, bg, line_type):

        y_pos = self.total_h - self.blank_size/2
        min_x, max_x = int(self.blank_size/2), int(self.total_w - self.blank_size/2)

        if line_type == "first":
            # first draw light line
            draw_line_in_loss = self.draw_line_in_loss
            while y_pos>(self.blank_size/2 - 1e-3):
                cv2.line(bg, (min_x, int(y_pos)), (max_x, int(y_pos)), (180,180,180), 1)
                y_pos -= self.h/(self.max_vis_loss*5*draw_line_in_loss)

        if line_type == "second":
            # draw black line (horizontal)
            point_count = 0
            while y_pos>(self.blank_size/2 - 1e-3):
                cv2.line(bg, (min_x, int(y_pos)), (max_x, int(y_pos)), (0,0,0), 1)
                msg = str(round(point_count, 1))
                d = self._txt_d(msg)
                cv2.putText(bg, msg, (int(min_x-15-d), int(y_pos)), 1, 0.8, (0,0,0), 1)
                y_pos -= self.h/(self.max_vis_loss*5)
                point_count += 0.2


    def plot_list(self, start_index, dlist, showval_list, dot_color=(200,0,0), xtype='batch', ytype='loss'):
        """
        input dlist_x and dlist_y to plot.
        """
        prvs_p, this_p = None, None
        for i in range(len(dlist)):
            this_p = self.data2plotpos(x=i+start_index, y=dlist[i], xtype=xtype, ytype=ytype)
            cv2.circle(self.background, this_p, 2, dot_color, -1)
            if showval_list[i]:
                msg = str(round(dlist[i], 1))
                cv2.putText(self.background, msg, (this_p[0]-4, this_p[1]-10), 1, 0.8, dot_color, 1)
            
            if prvs_p is not None:
                cv2.line(self.background, prvs_p, this_p, dot_color, 1)

            prvs_p = this_p

        
    def clear_background(self):
        self.background = self.plot_bg(self.batch_size, 
                                       self.sample_nums, 
                                       self.update_every_batches, 
                                       self.total_epoches, 
                                       self.mode)



"""
# a simple demonstration of how to use this.
# run 'python3 cv2.plot.py'
if __name__ == "__main__":
    
    # initial  
    canvas = Canvas(batch_size=32, 
                    sample_nums=60000, 
                    update_every_batches=200, 
                    total_epoches=90,
                    blank_size=65, 
                    epoch_pixel=20, 
                    max_vis_loss=6.4,
                    canvas_h=500,
                    x_ruler=4,
                    y_ruler=2
                    )
    
    cv2.namedWindow('bg', 0)
    cv2.imshow('bg', canvas.background)
    cv2.waitKey(0)

    # plot with data
    dlist = [4.2, 4.1, 3.8, 3.6, 3.3, 3.3, 2.8, 1.9, 1.0, 1.1, 1.0, 0.9]
    showval_list = [False]*len(dlist)
    showval_list[5] = True
    canvas.plot_list(dlist, showval_list)

    cv2.namedWindow('bg', 0)
    cv2.imshow('bg', canvas.background)
    cv2.waitKey(0)

    # clear bg
    canvas.clear_background()
    cv2.namedWindow('bg', 0)
    cv2.imshow('bg', canvas.background)
    cv2.waitKey(0)
"""


