// Copyright 2013 jrweizhang AT gmail.com. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package opencv

//#include "opencv.h"
//#cgo linux  pkg-config: opencv
//#cgo darwin pkg-config: opencv
//#cgo windows LDFLAGS: -lopencv_core248 -lopencv_imgproc248 -lopencv_photo248 -lopencv_highgui248 -lstdc++
import "C"
import (
	//"errors"
	//"log"
	"unsafe"
)

const INTER_LINEAR = C.CV_INTER_LINEAR

func Resize(src *IplImage, width, height, interpolation int) *IplImage {
	if width == 0 && height == 0 {
		panic("Width and Height cannot be 0 at the same time")
	}
	if width == 0 {
		ratio := float64(height) / float64(src.Height())
		width = int(float64(src.Width()) * ratio)
	} else if height == 0 {
		ratio := float64(width) / float64(src.Width())
		height = int(float64(src.Height()) * ratio)
	}

	dst := CreateImage(width, height, src.Depth(), src.Channels())
	C.cvResize(unsafe.Pointer(src), unsafe.Pointer(dst), C.int(interpolation))
	return dst
}
