/*
날짜 : 2020.04.08

내용 : 인식한 데이터를 10개의 버퍼에 순차적으로 저장 후 다음 인식 데이터가 이전 버퍼 10개 중 7개와 일치하면 출력함.
		인식한 데이터는 10번째 버퍼에 저장되고 첫번째 버퍼에 있던 데이터는 삭제됨.
		데이터는 텍스트 파일로 저장됨.

- by 황영준
*/


#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

void captureArtag()
{
	/* Aruco marker 인식을 위한 변수설정 */
	int dictionaryId = 10;

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	vector< int > ids;
	vector< vector< Point2f> > corners, rejected;
	vector<Point2f> marker_corner;
	int left, right, top, bottom;

	/* 모니터링 할 IP 갯수 설정*/

	int Num_of_IP = 2; // ★★★ 이후 mfc 에서 사용자의 설정에 맞게 변경되도록 할 부분임.

	// 딥러닝 모델 네트워크 연결
	Net net = readNet("model_0406.pb");

	if (net.empty()) {
		cerr << "Network load failed!" << endl;
	}

	/* 웹캠연결 */
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return;
	}

	float IP1_Rate[10] = { 0, };
	float IP1_Volume[10] = { 0, };
	float IP2_Rate[10] = { 0, };
	float IP2_Volume[10] = { 0, };

	/* 웹캠 화면 출력 부분*/
	Mat frame;
	while (true) {
		cap >> frame;

		if (frame.empty())
			break;

		//arucomarker 인식 및 테두리,ID 표시
		aruco::detectMarkers(frame, dictionary, corners, ids, detectorParams, rejected);
		//aruco::drawDetectedMarkers(frame, corners, ids);
		//aruco::drawDetectedMarkers(frame, corners);

		// 환자 정보를 저장하는 배열 
		float P_data[2][3] = { 0, };
		
		// marker 가 인식 될 때, maker ID 를 화면에 출력 (marker id 를 patient id 로 사용할 생각임)
		if (!ids.empty()) {			

			// 1차 ROI 영역 검출 ( 1차 ROI 영역 - LED 화면이 있는 부분 전체를 ROI 로 1차적으로 설정하는 부분 )
			Mat ROI;
			Mat rect_sample;
			Mat ROI_right, ROI_left;			

			for (int i = 0; i < ids.size(); i++)
			{
				// marker ID 검출

				int marker_id = ids[i];

				P_data[i][0] = marker_id;  //환자 정보 배열에 환자ID 저장

				marker_corner = corners[i];
				left = marker_corner[0].x;     //인식된 arucomarker 의 왼쪽 모서리 x좌표
				top = marker_corner[0].y;      //인식된 arucomarker 의 위쪽 모서리 y좌표
				right = marker_corner[2].x;    //인식된 arucomarker 의 오른쪽 모서리 x좌표
				bottom = marker_corner[2].y;   //인식된 arucomarker 의 오른쪽 모서리 y좌표


				// Arucomarker의 대각선 길이(코너0 ~ 코너2)를 구한다.
				// 카메라가 움직일 때, 이 대각선의 길이 비율대로 ROI 영역의 비율이 자동으로 바뀌게 됨.
				double d_arucomarker = sqrt(pow(right - left, 2) + pow(top - bottom, 2));
				double d_ROI = (230 * d_arucomarker) / 36;
				double width = 0.94*d_ROI;//0.87*d_ROI; //0.94*d_ROI;
				double height = 0.25*width; //0.2*width;//0.25*width;
				double roi_left = left - width;
				double roi_top = top - height;

				if (roi_left <= 0) {
					cout << "IP is out of range" << endl;
				}
				else {
					Rect rect(roi_left, roi_top, width, height);
					ROI = frame(rect);

					Mat gray;
					cvtColor(ROI, gray, COLOR_BGR2GRAY);

					Mat dst1;
					Canny(gray, dst1, 190, 200); //'gray' 에 대해 낮은 임계값을 190, 높은 임계값을 200 으로 설정하여 Canny Edge Detection 을 수행하고, 그 결과를 dst1에 저장.

					Mat labels, stats, centroids;
					int cnt = connectedComponentsWithStats(dst1, labels, stats, centroids); // 라벨링 하는 부분

					//Labelling 면적에 따라 큰 순서부터 작은 순서대로 stats 행을 재배열

					Mat stats_copy(cnt, 5, CV_8UC1);                              //내림차순으로 재배열할 때 임시로 저장할 행렬

					for (int i = 0; i < cnt - 1; i++) {
						for (int j = i + 1; j < cnt; j++) {
							if (stats.at<int>(i, 4) < stats.at<int>(j, 4)) {

								stats.row(j).copyTo(stats_copy);               //이부분에서 '깊은 복사'를 해야함.
								stats.row(i).copyTo(stats.row(j));
								stats_copy.copyTo(stats.row(i));

							}
						}
					}

					for (int j = 1; j < 3; j++) {

						int* p = stats.ptr<int>(j);

						if (p[4] < 50) continue;

						//rectangle(ROI, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255), 1);
						Mat rect_sample = ROI(Rect(p[0], p[1], p[2], p[3]));


						if (j == 1) { // 오른쪽 LED 화면

							float num_right[5] = { 0, }; // 숫자 인식 후 각각의 숫자를 저장할 배열 지정, 길이 5 & 초기값 전부 0 으로 설정

							for (int j = 0; j < 5; j++) {

								//int* numPtr[j];

								//rectangle(ROI, Rect(p[0] + 1 + p[2] * j / 5, p[1] + 1, p[2] / 5, p[3] - 1), 1);
								Mat rect_sample_right = ROI(Rect(p[0] + 1 + p[2] * j / 5, p[1] + 1, p[2] / 5, p[3] - 1));

								Mat rect_sample_right_resize;
								resize(rect_sample_right, rect_sample_right_resize, Size(224, 224), 0, 0, INTER_LINEAR);

								vector<Mat> bgr_planes;
								split(rect_sample_right_resize, bgr_planes);

								Mat BG_right(Size(224, 224), CV_8UC1);
								BG_right = 0.5*bgr_planes[0] + 0.5*bgr_planes[1];

								Mat blob = blobFromImage(BG_right, 1 / 255.f, Size(224, 224));
								net.setInput(blob);
								Mat prob = net.forward();

								double maxVal;
								Point maxLoc;
								minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
								float digit = maxLoc.x;

								if (digit == 10) {
									digit = 0;
								}

								num_right[j] = digit; // 인식한 각 숫자를 배열에 자리순서대로 저장

								/*int n1 = digit;
								char n2[10];
								sprintf_s(n2, "%d", n1);
								putText(BG_right, n2, Point(20, 40), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255));*/

								string Window_number_R;
								Window_number_R = format("Right_%d.jpg", j);
								imshow(Window_number_R, BG_right);
							}

							float sum_right = 1000 * num_right[0] + 100 * num_right[1] + 10 * num_right[2] + 1 * num_right[3] + 0.1 * num_right[4];

							waitKey(10);

							P_data[i][2] = sum_right;
						}

						else if (j == 2) { // 왼쪽 LED 화면

							float num_left[4] = { 0, }; // 숫자 인식 후 각각의 숫자를 저장할 배열 지정, 길이 5 & 초기값 전부 0 으로 설정

							for (int j = 0; j < 4; j++) {

								//rectangle(ROI, Rect(p[0] + 1 + p[2] * j / 4, p[1] + 1, p[2] / 4, p[3] - 2), 1);
								Mat rect_sample_left = ROI(Rect(p[0] + 1 + p[2] * j / 4, p[1] + 1, p[2] / 4, p[3] - 2));

								Mat rect_sample_left_resize;
								resize(rect_sample_left, rect_sample_left_resize, Size(224, 224), 0, 0, INTER_LINEAR);

								vector<Mat> bgr_planes;
								split(rect_sample_left_resize, bgr_planes);

								Mat BG_left(Size(224, 224), CV_8UC1);
								BG_left = 0.5*bgr_planes[0] + 0.5*bgr_planes[1];

								Mat blob = blobFromImage(BG_left, 1 / 255.f, Size(224, 224));
								net.setInput(blob);
								Mat prob = net.forward();

								double maxVal;
								Point maxLoc;
								minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
								float digit = maxLoc.x;

								if (digit == 10) {
									digit = 0;
								}

								num_left[j] = digit;  // 인식한 각 숫자를 배열에 자리순서대로 저장

								//int n1 = digit;
								//char n2[10];
								//sprintf_s(n2, "%d", n1);
								//putText(BG_left, n2, Point(20, 40), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 255));

								string Window_number_L;
								Window_number_L = format("left_%d.jpg", j);
								imshow(Window_number_L, BG_left);

							}

							float sum_left = 100 * num_left[0] + 10 * num_left[1] + 1 * num_left[2] + 0.1 * num_left[3];

							waitKey(10);

							P_data[i][1] = sum_left;

						}
					}
				}
			}
		}
		else {
			cout << "There is no IP" << endl;
		}		

		int j, k;
		int cnt_IP1_rate = 0;
		int cnt_IP1_volume = 0;
		int cnt_IP2_rate = 0;
		int cnt_IP2_volume = 0;
			   
		if (P_data[1][0] > P_data[0][0]) {

			for (j = 0; j < 10; j++) {

				if (IP1_Rate[j] == P_data[0][1])
					cnt_IP1_rate++;
				if (IP1_Volume[j] == P_data[0][2])
					cnt_IP1_volume++;
				if (IP2_Rate[j] == P_data[1][1])
					cnt_IP2_rate++;
				if (IP2_Volume[j] == P_data[1][2])
					cnt_IP2_volume++;

			}

			if (cnt_IP1_rate > 6 && cnt_IP1_volume > 6 && cnt_IP2_rate > 6 && cnt_IP2_volume > 6) {

				//cout << "*******************" << endl;
				cout << "#" << " " << P_data[0][0] << endl;
				cout << P_data[0][1] << endl;
				cout << P_data[0][2] << endl;
				cout << " " << endl;
				cout << "#" << " " << P_data[1][0] << endl;
				cout << P_data[1][1] << endl;
				cout << P_data[1][2] << endl;
				cout << "*******************" << endl;

				ofstream output("output.txt", ios::app);

				output << P_data[0][0] << " " << P_data[0][1] << " " << P_data[0][2] << " " << P_data[1][0] << " " << P_data[1][1] << " " << P_data[1][2] << endl;

				output.close();

			}

			for (k = 0; k < 9; k++) {

				IP1_Rate[k] = IP1_Rate[k + 1];
				IP1_Volume[k] = IP1_Volume[k + 1];
				IP2_Rate[k] = IP2_Rate[k + 1];
				IP2_Volume[k] = IP2_Volume[k + 1];

			}					

			IP1_Rate[9] = P_data[0][1];
			IP1_Volume[9] = P_data[0][2];
			IP2_Rate[9] = P_data[1][1];
			IP2_Volume[9] = P_data[1][2];

		}
		else {

			for (j = 0; j < 10; j++) {

				if (IP1_Rate[j] == P_data[1][1])
					cnt_IP1_rate++;
				if (IP1_Volume[j] == P_data[1][2])
					cnt_IP1_volume++;
				if (IP2_Rate[j] == P_data[0][1])
					cnt_IP2_rate++;
				if (IP2_Volume[j] == P_data[0][2])
					cnt_IP2_volume++;

			}

			if (cnt_IP1_rate > 6 && cnt_IP1_volume > 6 && cnt_IP2_rate > 6 && cnt_IP2_volume > 6) {

				//cout << "*******************" << endl;
				cout << "#" << " " << P_data[1][0] << endl;
				cout << P_data[1][1] << endl;
				cout << P_data[1][2] << endl;
				cout << " " << endl;
				cout << "#" << " " << P_data[0][0] << endl;
				cout << P_data[0][1] << endl;
				cout << P_data[0][2] << endl;
				cout << "*******************" << endl;

				ofstream output("output.txt", ios::app);

				output << P_data[1][0] << " " << P_data[1][1] << " " << P_data[1][2] << " " << P_data[0][0] << " " << P_data[0][1] << " " << P_data[0][2] << endl;

				output.close();

			}

			IP1_Rate[9] = P_data[1][1];
			IP1_Volume[9] = P_data[1][2];
			IP2_Rate[9] = P_data[0][1];
			IP2_Volume[9] = P_data[0][2];

			if (ids.size() < Num_of_IP) {
				cout << "One infusion pump is out of range." << endl;
				//cout << "*******************" << endl;
				cout << "#" << " " << P_data[1][0] << endl;
				cout << P_data[1][1] << endl;
				cout << P_data[1][2] << endl;
				cout << " " << endl;
				cout << "#" << " " << P_data[0][0] << endl;
				cout << P_data[0][1] << endl;
				cout << P_data[0][2] << endl;
				cout << "*******************" << endl;
			}

		}
		
		imshow("frame", frame);

		if (waitKey(100) == 27) // 'ESC' key
			break;
	}
}

int main()
{

	captureArtag();

	return 0;

}
