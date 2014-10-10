// Copyright 2011 <chaishushan@gmail.com>. All rights reserved.
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
	"fmt"
	"unsafe"
)

const (
	CV_BGR2GRAY  = C.CV_BGR2GRAY
	CV_BGR2BGRA  = C.CV_BGR2BGRA
	CV_RGBA2BGRA = C.CV_RGBA2BGRA

	CV_BLUR     = C.CV_BLUR
	CV_GAUSSIAN = C.CV_GAUSSIAN

	CV_8U  = C.CV_8U
	CV_8S  = C.CV_8S
	CV_16U = C.CV_16U
	CV_16S = C.CV_16S
	CV_32S = C.CV_32S
	CV_32F = C.CV_32F
	CV_64F = C.CV_64F
)

func Laplace(src, dst *IplImage, n int) {
	C.cvLaplace(unsafe.Pointer(src), unsafe.Pointer(dst), C.int(n))
}

const HOUGH_GRADIENT = C.CV_HOUGH_GRADIENT

func HoughCircles(
	img *IplImage, method int,
	dp, mindist, param1, param2 float64,
	min_radius, max_radius int) []float32 {

	store := C.cvCreateMemStorage(0)

	seq := C.cvHoughCircles(unsafe.Pointer(img), unsafe.Pointer(store), C.int(method),
		C.double(dp), C.double(mindist),
		C.double(param1), C.double(param2),
		C.int(min_radius), C.int(max_radius))

	fmt.Println("n.o. Cicles: ", seq.total)
	ar := make([]float32, int(seq.total)*3)
	C.cvCvtSeqToArray(seq, unsafe.Pointer(&ar[0]), C.cvSlice(0, 0x3fffffff))
	return ar
}

// int cvFindContours(
// 	CvArr* image,
// 	CvMemStorage* storage,
// 	CvSeq** first_contour,
// 	int header_size=sizeof(CvContour),
// 	int mode=CV_RETR_LIST,
// 	int method=CV_CHAIN_APPROX_SIMPLE,
// 	CvPoint offset=cvPoint(0,0)
// 	)

const (
	RETR_EXTERNAL = C.CV_RETR_EXTERNAL
	RETR_LIST     = C.CV_RETR_LIST
	RETR_CCOMP    = C.CV_RETR_CCOMP
	RETR_TREE     = C.CV_RETR_TREE
)

const (
	CHAIN_APPROX_NONE      = C.CV_CHAIN_APPROX_NONE
	CHAIN_APPROX_SIMPLE    = C.CV_CHAIN_APPROX_SIMPLE
	CHAIN_APPROX_TC89_L1   = C.CV_CHAIN_APPROX_TC89_L1
	CHAIN_APPROX_TC89_KCOS = C.CV_CHAIN_APPROX_TC89_KCOS
)

// type Point struct{ x, y int }
// type Contour []Point

// func FindContours(img *IplImage, mode, method int, e float64) [][]Point {
// 	store := C.cvCreateMemStorage(0)
// 	seq := new(C.CvSeq)
// 	header_size := C.int(unsafe.Sizeof(C.CvContour{}))
// 	num_polys := C.cvFindContours(unsafe.Pointer(img), store, &seq, header_size,
// 		C.int(mode), C.int(method), C.cvPoint(C.int(0), C.int(0)))

// 	if num_polys == 0 {
// 		return [][]Point{}
// 	}

// 	store2 := C.cvCreateMemStorage(0)
// 	seq2 := C.cvApproxPoly(unsafe.Pointer(seq), header_size, store2, C.CV_POLY_APPROX_DP, C.double(e), C.int(1))

// 	seq = seq2

// 	result := [][]Point{}

// 	for {
// 		ar := make([]Point, seq.total)

// 		C.cvCvtSeqToArray(seq, unsafe.Pointer(&ar[0]), C.cvSlice(0, 0x3fffffff))
// 		result = append(result, ar)

// 		if seq.h_next == nil {
// 			break
// 		}
// 		tmp := C.CvSeq(*seq.h_next)
// 		seq = &tmp
// 	}

// 	return result
// }

type ContourTree struct {
	Data [][]Point
	Next []*ContourTree
}

func FindContours(img *IplImage, mode, method int, e float64) *ContourTree {
	store := C.cvCreateMemStorage(0)
	seq := new(C.CvSeq)
	header_size := C.int(unsafe.Sizeof(C.CvContour{}))
	num_polys := C.cvFindContours(unsafe.Pointer(img), store, &seq, header_size,
		C.int(mode), C.int(method), C.cvPoint(C.int(0), C.int(0)))

	if num_polys == 0 {
		return &ContourTree{}
	}

	store2 := C.cvCreateMemStorage(0)
	seq2 := C.cvApproxPoly(unsafe.Pointer(seq), header_size, store2, C.CV_POLY_APPROX_DP, C.double(e), C.int(1))

	return MakeTree(seq2)
}

func MakeTree(seq *C.CvSeq) *ContourTree {
	if seq == nil {
		return nil
	}

	t := &ContourTree{}
	for ; seq != nil; seq = (*C.CvSeq)(seq.h_next) {
		ar := make([]Point, seq.total)
		C.cvCvtSeqToArray(seq, unsafe.Pointer(&ar[0]), C.cvSlice(0, 0x3fffffff))
		t.Data = append(t.Data, ar)
		t.Next = append(t.Next, MakeTree((*C.CvSeq)(seq.v_next)))
	}
	return t
}

func Erode(src, dst *IplImage, element *IplConvKernel, iterations int) {
	// e := C.IplConvKernel(*element)
	// f := &e
	// f = nil
	element = nil
	var f *C.IplConvKernel = nil
	C.cvErode(unsafe.Pointer(src), unsafe.Pointer(dst), f, C.int(iterations))
}

func Dilate(src, dst *IplImage, element *IplConvKernel, iterations int) {
	// e := C.IplConvKernel(*element)
	// f := &e
	// f = nil
	element = nil
	var f *C.IplConvKernel = nil
	C.cvDilate(unsafe.Pointer(src), unsafe.Pointer(dst), f, C.int(iterations))
}

// func Resize(src, dst *IplImage) {
// 	C.cvResize(unsafe.Pointer(src), unsafe.Pointer(dst), C.CV_INTER_LINEAR)
// }

/* Smoothes array (removes noise) */
func Smooth(src, dst *IplImage, smoothtype,
	param1, param2 int, param3, param4 float64) {
	C.cvSmooth(unsafe.Pointer(src), unsafe.Pointer(dst), C.int(smoothtype),
		C.int(param1), C.int(param2), C.double(param3), C.double(param4),
	)
}

const CV_THRESH_BINARY = C.CV_THRESH_BINARY

// 	THRESH_BINARY_INV = C.THRESH_BINARY_INV
// 	THRESH_TRUNC      = C.THRESH_TRUNC
// 	THRESH_TOZERO     = C.THRESH_TOZERO
// 	THRESH_TOZERO_INV = C.THRESH__INV
// )

func Threshold(src, dst *IplImage, threshold, max_value float64, threshold_type int) {
	C.cvThreshold(unsafe.Pointer(src), unsafe.Pointer(dst), C.double(threshold), C.double(max_value), C.int(threshold_type))
}

/*
ConvertScale converts one image to another with optional linear transformation.
*/
func ConvertScale(a, b *IplImage, scale, shift float64) {
	C.cvConvertScale(unsafe.Pointer(a), unsafe.Pointer(b), C.double(scale), C.double(shift))
}

//CVAPI(void)  cvConvertScale( const CvArr* src,
//                             CvArr* dst,
//                             double scale CV_DEFAULT(1),
//                             double shift CV_DEFAULT(0) );

/* Converts input array pixels from one color space to another */
func CvtColor(src, dst *IplImage, code int) {
	C.cvCvtColor(unsafe.Pointer(src), unsafe.Pointer(dst), C.int(code))
}

//CVAPI(void)  cvCvtColor( const CvArr* src, CvArr* dst, int code );

/* Runs canny edge detector */
func Canny(image, edges *IplImage, threshold1, threshold2 float64, aperture_size int) {
	C.cvCanny(unsafe.Pointer(image), unsafe.Pointer(edges),
		C.double(threshold1), C.double(threshold2),
		C.int(aperture_size),
	)
}

//CVAPI(void)  cvCanny( const CvArr* image, CvArr* edges, double threshold1,
//                      double threshold2, int  aperture_size CV_DEFAULT(3) );

const (
	CV_INPAINT_NS    = C.CV_INPAINT_NS
	CV_INPAINT_TELEA = C.CV_INPAINT_TELEA
)

/* Inpaints the selected region in the image */
func Inpaint(src, inpaint_mask, dst *IplImage, inpaintRange float64, flags int) {
	C.cvInpaint(
		unsafe.Pointer(src),
		unsafe.Pointer(inpaint_mask),
		unsafe.Pointer(dst),
		C.double(inpaintRange),
		C.int(flags),
	)
}

//CVAPI(void) cvInpaint( const CvArr* src, const CvArr* inpaint_mask,
//                       CvArr* dst, double inpaintRange, int flags );
