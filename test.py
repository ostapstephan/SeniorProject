#!/usr/bin/env python
import pbcvt
import cv2

# TRUTH = (194.015831, 203.153840, 46.952739, 22.625909, -0.529757)


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=3, lineType=cv2.LINE_AA, shift=10):
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)


cv2.namedWindow('test')
xx = 0
while(True):
    img = cv2.imread('data/render_eye_'+str(xx % 49)+'.png', cv2.IMREAD_COLOR)
    xx += 1

    out = pbcvt.findPupilEllipse(img)
    # cv2.ellipse(img, out, (0, 255, 0), 2)
    draw_ellipse(img, (out[0], out[1]), (out[2]/2, out[3]/2), out[4],
                 0, 360, (0, 0, 0), 2)
    # draw_ellipse(img, (TRUTH[0], TRUTH[1]), (TRUTH[2], TRUTH[3]), TRUTH[4],
    #              0, 360, (0, 0, 0), 2)
    cv2.imshow('test', img)

    print(out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
