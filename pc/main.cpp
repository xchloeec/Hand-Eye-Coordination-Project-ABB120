// Hand eye coordination - PC side (RME40003)
// Detects the blocks, converts camera positions to robot XY, and sends the stack/return plans.
// MainModule.mod runs the robot. The camera checks each transfer while the robot waits at HOME.
// Static uses P1-P9; Dynamic uses camera targets; Calibrate sets the mapping; Test checks a pick.
//
// Serial format (PC to robot, no reply)
//    Each value is held on the robot's 8 bit group input "dg" for 250 ms, with a 0 between values:
//        0, START, 0, field1, 0, field2, ... 0, END, 0
//    Frames are at least 1.5 s apart so the controller is listening again before the next start byte.
//    Checksum = (sum of the fields) mod 251 + 1. XY is sent in 0.5 mm steps:
//        X = 100 + q/2,  Y = -400 + q/2,  q = (hi - 1) * 128 + (lo - 1)
//      226..229          mode select (static, dynamic, test, calibrate)
//      240 .. 241        static plan: stack site, L1, L2, L3 pickup numbers
//      220 .. 221        dynamic stack plan: token, pattern, 3 x XY, checksum
//      222 .. 223        dynamic return plan: token, 3 x XY, checksum
//      234 .. 235        control: token, action (1 HOLD, 2 HOME, 3 RESET, 4 PERMIT), step, checksum
//      236 .. 237        camera target: token, mode (1 re-pick, 2 move next target), step, attempt,
//                        XY, checksum
//      232 .. 233        TEST page: token, action (hover, pick, cancel), XY, checksum
//    The token identifies one cycle, so the robot can refuse stale or repeated frames.
//
// v10-v13 prefixes come from earlier project versions and match the robot module.
//
// Build and run
//    C++17, OpenCV 4, libserialport.  Run:  HandEye.exe COM3
//    Self-tests (no camera or robot needed): --self-test, --v10-self-test, --v13-self-test,
//    --calibration-self-test, --ui-audit-self-test and the others listed in main().
//
// Cell safety
//    Software checks here (workspace limits, clearance, camera checks) are not safety functions.
//    The robot controller STOP and the guarding of the cell are the safety measures.
//
// Where to find things (search for "SECTION")
//     1  Calibration and detection (namespace calibrationui)
//     2  Shared state
//     3  Small helpers, settings file, static mode vision and planning
//     4  Serial link
//     5  Stack monitor and automatic lighting
//     6  Static mode operator actions
//     7  Pick offsets and pick face checks
//     8  Static build and TEST page actions
//     9  Dynamic mode: workspace, plans and permits
//    10  Dynamic mode: camera supervisor
//    11  Dynamic mode: main state machine
//    12  Dashboard drawing and mouse input
//    13  Self-tests and previews
//    14  Program entry and main loop
// ------------------------------------------------------------
#include <opencv2/opencv.hpp>
#include <libserialport.h>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <bitset>
#include <sstream>
#include <iomanip>
#include <limits>
#include <random>
#include <chrono>
#include <cerrno>
#include <stdexcept>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif
#include <fstream>
#include <cassert>
#include <ctime>
#include <numeric>
#include <deque>
#include <functional>
#include <thread>
#include <atomic>
using namespace cv;
using namespace std;

// ------------------------------------------------------------
// SECTION 1  CALIBRATION AND DETECTION  (namespace calibrationui)
//    Everything that turns a camera image into block positions: colour thresholds and detection,
//    top-face estimation, the P1 to P9 homography, and the CALIBRATE page itself.
// ------------------------------------------------------------
namespace calibrationui {
    // C++14 or newer / OpenCV core,imgproc,imgcodecs,videoio,highgui only.
    // Integrated calibration + coordinate test + planning preview.
    // Three stack patterns are retained from the backed-up planning main.cpp.
    // No serial library, command sender or robot motion.
#ifdef _WIN32
#else
#endif
    namespace fileutil {
        using namespace std;
        string currentPath() {
            char buffer[32768];
#ifdef _WIN32
            if (!_getcwd(buffer, sizeof(buffer))) throw runtime_error("cannot read working directory");
#else
            if (!getcwd(buffer, sizeof(buffer))) throw runtime_error("cannot read working directory");
#endif
            return buffer;
        }
        void currentPath(const string& path) {
#ifdef _WIN32
            if (_chdir(path.c_str()) != 0) throw runtime_error("cannot change working directory: " + path);
#else
            if (chdir(path.c_str()) != 0) throw runtime_error("cannot change working directory: " + path);
#endif
        }
        void createDirectory(const string& path) {
#ifdef _WIN32
            int result = _mkdir(path.c_str());
#else
            int result = mkdir(path.c_str(), 0777);
#endif
            if (result != 0) {
                struct stat info;
                if (errno != EEXIST || stat(path.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR))
                    throw runtime_error("cannot create directory: " + path);
            }
        }
        bool exists(const string& path) {
            ifstream file(path, ios::binary);
            return file.good();
        }
    }

    using cv::Point2d;
    using H = cv::Matx33d;
    constexpr int UIW = 1280, UIH = 900, STABLE_FRAMES = 20; // Collect 20 steady frames for calibration.
    constexpr double RESET_JUMP_PX = 12.0, DRIFT_LIMIT_PX = 3.0; // relaxed: fit gate (2 mm) is still enforced
    const cv::Rect VIDEO(15, 100, 820, 615);
    const cv::Scalar BG(30, 26, 23), PANEL(48, 43, 38), FG(240, 240, 235), MUTED(185, 180, 170), GOLD(85, 205, 245);
    const char* TITLE = "Block Calibration | Preview Only | NO ROBOT CONNECTION";
    std::string fmt(double v, int n = 2) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(n) << v;
        return s.str();
    }
    std::string stamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char b[40];
        std::strftime(b, sizeof(b), "%Y%m%d_%H%M%S", &tm);
        return b;
    }
    Point2d project(const H& h, Point2d p) {
        cv::Vec3d v = h * cv::Vec3d(p.x, p.y, 1);
        if (std::abs(v[2]) < 1e-10) throw std::runtime_error("Mapping denominator is zero");
        return { v[0] / v[2], v[1] / v[2] }; // Convert the mapped point back to ordinary XY coordinates.
    }
    std::vector<Point2d> localGrid() {
        std::vector<Point2d> p;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) p.emplace_back(40 + 75 * c, 180 - 70 * r);
        return p;
    }
    std::vector<Point2d> nominalRobot() {
        auto p = localGrid();
        for (auto& a : p) a = { 552.65 - a.y, a.x - 122.80 }; // Match the grid axes and origin to the taught robot positions.
        return p;
    }
    double polygonArea(const std::vector<Point2d>& p) {
        std::vector<cv::Point2f> q;
        for (auto a : p) q.emplace_back(float(a.x), float(a.y));
        std::vector<cv::Point2f> hull;
        cv::convexHull(q, hull);
        return std::abs(cv::contourArea(hull));
    }
    bool fitH(const std::vector<Point2d>& src, const std::vector<Point2d>& dst, H& result) {
        if (src.size() < 4 || src.size() != dst.size() || polygonArea(src) < 10 || polygonArea(dst) < 10) return false;
        auto norm = [](const std::vector<Point2d>& p) {
            Point2d mean{};
            for (auto v : p) mean += v;
            mean *= 1. / p.size();
            double d = 0;
            for (auto v : p) d += cv::norm(v - mean);
            d /= p.size();
            double s = std::sqrt(2.) / std::max(d, 1e-12);
            return H(s, 0, -s * mean.x, 0, s, -s * mean.y, 0, 0, 1);
            };
        H a = norm(src), b = norm(dst);
        cv::Mat m(int(src.size() * 2), 9, CV_64F, cv::Scalar(0));
        for (size_t i = 0; i < src.size(); ++i) {
            auto u = project(a, src[i]), x = project(b, dst[i]);
            double r0[] = { -u.x, -u.y, -1, 0, 0, 0, x.x * u.x, x.x * u.y, x.x };
            double r1[] = { 0, 0, 0, -u.x, -u.y, -1, x.y * u.x, x.y * u.y, x.y };
            for (int j = 0; j < 9; ++j) {
                m.at<double>(int(i * 2), j) = r0[j];
                m.at<double>(int(i * 2 + 1), j) = r1[j];
            }
        }
        cv::SVD svd(m, cv::SVD::FULL_UV);
        if (svd.w.rows < 8 || svd.w.at<double>(7) < svd.w.at<double>(0) * 1e-7) return false;
        H h;
        for (int i = 0; i < 9; ++i) h.val[i] = svd.vt.at<double>(8, i);
        h = b.inv() * h * a;
        if (std::abs(h(2, 2)) < 1e-10) return false;
        h *= 1. / h(2, 2);
        if (!cv::checkRange(cv::Mat(h))) return false;
        result = h;
        return true;
    }
    // Robust seed only: every final calibration point must still be recovered
    // from current image pixels and included in the final nine-point fit.
    bool fitGridSeed(const std::vector<Point2d>& pixels, const std::vector<Point2d>& refs, H& result) {
        if (pixels.size() != 9 || refs.size() != 9) return false;
        H initial;
        if (fitH(pixels, refs, initial)) {
            double worst = 0;
            for (int i = 0; i < 9; ++i) worst = std::max(worst, cv::norm(project(initial, pixels[i]) - refs[i]));
            if (worst <= 3) {
                result = initial;
                return true;
            }
        }
        int bestCount = 0;
        double bestError = 1e9;
        H best;
        for (int a = 0; a < 9; ++a)
            for (int b = a + 1; b < 9; ++b) {
                std::vector<Point2d> src, dst;
                for (int i = 0; i < 9; ++i)
                    if (i != a && i != b) {
                        src.push_back(pixels[i]);
                        dst.push_back(refs[i]);
                    }
                H h;
                if (!fitH(src, dst, h)) continue;
                int count = 0;
                double error = 0;
                for (int i = 0; i < 9; ++i) {
                    double e = cv::norm(project(h, pixels[i]) - refs[i]);
                    if (e <= 3) ++count;
                    error += std::min(9., e * e);
                }
                if (count > bestCount || (count == bestCount && error < bestError)) {
                    bestCount = count;
                    bestError = error;
                    best = h;
                }
            }
        if (bestCount < 6) return false;
        std::vector<Point2d> src, dst;
        for (int i = 0; i < 9; ++i)
            if (cv::norm(project(best, pixels[i]) - refs[i]) <= 3) {
                src.push_back(pixels[i]);
                dst.push_back(refs[i]);
            }
        return fitH(src, dst, result);
    }
    struct LightingInfo {
        double gain = 1;
        int reference = 0;
        std::string state = "STARTING";
        std::string text() const {
            std::ostringstream s;
            s << "LIGHT MONITOR " << std::fixed << std::setprecision(2) << gain << "x / " << state;
            return s.str();
        }
    };
    cv::Mat lightingMonitoredHSV(const cv::Mat& image, LightingInfo* info = nullptr) {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        std::array<int, 256> neutral{}, all{};
        int n = 0, total = 0;
        for (int y = 0; y < hsv.rows; y += 4)
            for (int x = 0; x < hsv.cols; x += 4) {
                auto p = hsv.at<cv::Vec3b>(y, x);
                ++all[p[2]];
                ++total;
                if (p[1] < 55 && p[2] >= 15) {//changed from 12 to 15
                    ++neutral[p[2]];
                    ++n;
                }
            }
        const auto& hist = n >= 200 ? neutral : all;
        int target = int(.70 * (n >= 200 ? n : total)), sum = 0, reference = 1;
        for (int i = 0; i < 256; ++i) {
            sum += hist[i];
            if (sum >= target) {
                reference = std::max(1, i);
                break;
            }
        }
        // Rollback: monitor exposure only. Detection receives EXACT raw HSV as before auto light.
        // Do not let a global background reference alter thresholds on dim coloured faces.
        if (info) {
            info->gain = 1;
            info->reference = reference;
            info->state = reference < 35 ? "TOO DARK"
                : reference < 80 ? "DIM"
                : reference > 245 ? "VERY BRIGHT"
                : "RAW / OK";
        }
        return hsv;
    }
    struct ColourModel {
        bool staticStyle = false, greenRescue = false;
        int sat = 55, val = 35, minArea = 200, coherenceKernel = 0;
        std::array<int, 3> hue{ 0, 60, 110 }, tol{ 13, 25, 20 };
    };
    // Operator tuning (Calibration setup > Lighting > TUNING). One set per lighting profile
    // (0 LIGHT ON, 1 LIGHT OFF). -1 = use the calibration value. Saved in detection_tuning.yml.
    struct DetectionTuning {
        std::array<int, 2> satMin{ {-1, -1} }, valMin{ {-1, -1} };
        double matchLimit = 15; // largest lighting-reference difference accepted as a match
        int active = 0;         // lighting profile in use
        int sat() const { return satMin[active == 1 ? 1 : 0]; }
        int val() const { return valMin[active == 1 ? 1 : 0]; }
    };
    inline DetectionTuning& detectionTuning() {
        static DetectionTuning t;
        return t;
    }
    cv::Mat colourThreshold(const cv::Mat& hsv, int k, const ColourModel& cfg) {
        cv::Mat mask(hsv.size(), CV_8U, cv::Scalar(0)), part;
        if (cfg.staticStyle && k == 0 && cfg.hue[0] == 0) {
            cv::inRange(hsv, cv::Scalar(0, 100, 80), cv::Scalar(10, 255, 255), mask);
            cv::inRange(hsv, cv::Scalar(170, 100, 80), cv::Scalar(179, 255, 255), part);
            mask |= part;
        }
        else if (cfg.staticStyle && k == 1 && cfg.hue[1] == 60)
            cv::inRange(hsv, cv::Scalar(35, 70, 60), cv::Scalar(85, 255, 255), mask);
        else if (cfg.staticStyle && k == 2 && cfg.hue[2] == 110)
            cv::inRange(hsv, cv::Scalar(90, 80, 60), cv::Scalar(130, 255, 255), mask);
        else
            for (int shift = -180; shift <= 180; shift += 180) {
                int lo = std::max(0, cfg.hue[k] - cfg.tol[k] + shift),
                    hi = std::min(179, cfg.hue[k] + cfg.tol[k] + shift);
                if (lo <= hi) {
                    cv::inRange(hsv, cv::Scalar(lo, cfg.sat, cfg.val), cv::Scalar(hi, 255, 255), part);
                    mask |= part;
                }
            }
        if (k == 1 && cfg.greenRescue) {
            // Green paint can lose saturation in diffuse light. Current-frame only;
            // neutral pixels remain excluded and every component still passes geometric gates.
            int lo = cfg.hue[1] == 60 ? 35 : std::max(0, cfg.hue[1] - cfg.tol[1]);
            int hi = cfg.hue[1] == 60 ? 85 : std::min(179, cfg.hue[1] + cfg.tol[1]);
            const auto& tune = detectionTuning();
            int rescueSat = tune.sat() >= 0 ? std::min(35, tune.sat()) : 35,
                rescueVal = tune.val() >= 0 ? std::min(25, tune.val()) : 25;
            cv::inRange(hsv, cv::Scalar(lo, rescueSat, rescueVal), cv::Scalar(hi, 255, 255), part);
            mask |= part;
        }
        return mask;
    }
    void cleanColourMask(cv::Mat& mask, bool staticStyle) {
        if (staticStyle) {
            auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 5, 5 });
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        }
        else {
            cv::medianBlur(mask, mask, 3);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, { 5, 5 }));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, { 3, 3 }));
        }
    }
    struct Detection {
        Point2d centre;
        cv::Rect box;
        std::vector<cv::Point> quad;
        int colour = 0;
        double area = 0;
        bool pickFaceVerified = true;
        // V13: robot-Y extent (mm) of the whole same-colour outline (top + any visible side). NaN = not measured.
        double blobMinY = std::numeric_limits<double>::quiet_NaN(), blobMaxY = std::numeric_limits<double>::quiet_NaN();
    };
    Point2d crossCentre(const std::vector<cv::Point>& q) {
        Point2d a = q[0], b = q[2], c = q[1], d = q[3], r = b - a, s = d - c;
        double den = r.x * s.y - r.y * s.x;
        if (std::abs(den) < 1e-9) return (a + b + c + d) * .25;
        double t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / den;
        return a + t * r;
    }
    struct DetectionDiagnostics {
        std::array<cv::Mat, 3> thresholdMasks, cleanMasks, fallbackThresholdMasks, fallbackCleanMasks;
        std::vector<std::string> rows;
    };
    // Finds coloured block faces in one image: threshold each colour in HSV, clean the mask,
    // keep block sized contours and fit a four corner face to each.
    std::vector<Detection> detect(const cv::Mat& image, const ColourModel& cfgIn,
        DetectionDiagnostics* diagnostics = nullptr, LightingInfo* lighting = nullptr) {
        // The adjustable S/V minimums of the active lighting replace the generic (dim) thresholds.
        // The fixed Static first pass is unchanged; its dim fallback below runs through this path.
        ColourModel cfg = cfgIn;
        if (!cfg.staticStyle) {
            const auto& tune = detectionTuning();
            if (tune.sat() >= 0) cfg.sat = tune.sat();
            if (tune.val() >= 0) cfg.val = tune.val();
        }
        cv::Mat prepared;
        if (cfg.staticStyle)
            cv::GaussianBlur(image, prepared, { 5, 5 }, 0);
        else
            prepared = image;
        cv::Mat hsv = lightingMonitoredHSV(prepared, lighting);
        std::vector<Detection> out;
        for (int k = 0; k < 3; ++k) {
            if (cfg.coherenceKernel > 0 && k != 1) continue;
            cv::Mat mask = colourThreshold(hsv, k, cfg);
            if (diagnostics) diagnostics->thresholdMasks[k] = mask.clone();
            cleanColourMask(mask, cfg.staticStyle);
            if (cfg.coherenceKernel > 0)
                cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                    cv::getStructuringElement(cv::MORPH_RECT, { cfg.coherenceKernel, cfg.coherenceKernel }));
            if (diagnostics) diagnostics->cleanMasks[k] = mask.clone();
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (auto& contour : contours) {
                double area = cv::contourArea(contour);
                auto report = [&](const std::string& reason, double solidity = 0) {
                    if (!diagnostics || area < 20) return;
                    auto r = cv::boundingRect(contour);
                    std::ostringstream row;
                    row << k << ',' << r.x << ',' << r.y << ',' << r.width << ',' << r.height << ',' << area << ','
                        << solidity << ',' << reason;
                    diagnostics->rows.push_back(row.str());
                    };
                if (area < cfg.minArea) {
                    report("too_small");
                    continue;
                }
                std::vector<cv::Point> hull, q;
                cv::convexHull(contour, hull);
                double hullArea = cv::contourArea(hull);
                // Fill small segmentation notches, but reject concave/merged shapes.
                if (hullArea <= 0 || area / hullArea < .85) {
                    report("low_solidity_or_merged", hullArea > 0 ? area / hullArea : 0);
                    continue;
                }
                // Rounded/blurred square corners reduce quad coverage to about 0.78-0.85.
                // Require convexity/solidity and edge aspect too; do not force rectangles onto arbitrary contours.
                bool quad = false;
                for (double epsilon : {.018, .025, .035, .045, .055, .065, .075}) {
                    cv::approxPolyDP(hull, q, cv::arcLength(hull, true) * epsilon, true);
                    if (q.size() == 4 && cv::isContourConvex(q)) {
                        double qa = std::abs(cv::contourArea(q));
                        if (qa / hullArea >= .75 && qa / hullArea <= 1.05) {
                            quad = true;
                            break;
                        }
                    }
                }
                if (!quad) {
                    // Bevelled corners can give 5-8 vertices although the coloured face is rectangular.
                    // Use a fitted rectangle only with high contour AND hull coverage, excluding circles.
                    auto rect = cv::minAreaRect(contour);
                    double rectArea = rect.size.area();
                    double shortSide = std::min(rect.size.width, rect.size.height),
                        longSide = std::max(rect.size.width, rect.size.height);
                    if (rectArea > 0 && shortSide >= 8 && longSide / shortSide <= 1.6 && area / rectArea >= .85 &&
                        hullArea / rectArea >= .88) {
                        cv::Point2f corners[4];
                        rect.points(corners);
                        q.clear();
                        for (auto p : corners) q.emplace_back(cvRound(p.x), cvRound(p.y));
                        quad = true;
                    }
                }
                if (!quad) {
                    report("not_quadrilateral", area / hullArea);
                    continue;
                }
                double minEdge = 1e9, maxEdge = 0;
                for (int j = 0; j < 4; ++j) {
                    double e = cv::norm(q[j] - q[(j + 1) % 4]);
                    minEdge = std::min(minEdge, e);
                    maxEdge = std::max(maxEdge, e);
                }
                if (minEdge < 8 || maxEdge / minEdge > 2.0) {
                    report("edge_size_or_aspect", area / hullArea);
                    continue;
                }
                cv::Rect box = cv::boundingRect(q);
                if (box.x <= 1 || box.y <= 1 || box.br().x >= image.cols - 1 || box.br().y >= image.rows - 1) {
                    report("image_border", area / hullArea);
                    continue;
                }
                if (area < box.area() * .45) {
                    report("low_fill", area / hullArea);
                    continue;
                }
                report("accepted", area / hullArea);
                out.push_back({ crossCentre(q), box, q, k, area });
            }
        }
        if (cfg.staticStyle) {
            // Static's high V/S cutoffs can fragment dim paint. A same-frame fallback retains
            // the recorded dim-block contour checks, never a held or invented detection.
            ColourModel dim = cfg;
            dim.staticStyle = false;
            dim.greenRescue = false;
            dim.sat = std::min(dim.sat, 55);
            dim.val = std::min(dim.val, 35);
            // Smaller distant top faces retain the same solidity/shape/border gates.
            dim.minArea = std::max(100, int(cfg.minArea * .60));
            const int defaults[3] = { 0, 60, 110 }, tolerance[3] = { 13, 25, 20 };
            for (int k = 0; k < 3; ++k)
                if (dim.hue[k] == defaults[k]) dim.tol[k] = std::max(dim.tol[k], tolerance[k]);
            DetectionDiagnostics fallback;
            auto extra = detect(image, dim, diagnostics ? &fallback : nullptr);
            if (diagnostics) {
                diagnostics->fallbackThresholdMasks = fallback.thresholdMasks;
                diagnostics->fallbackCleanMasks = fallback.cleanMasks;
                for (auto row : fallback.rows) {
                    auto at = row.rfind(',');
                    if (at != std::string::npos) row.insert(at + 1, "fallback_");
                    diagnostics->rows.push_back(row);
                }
            }
            for (auto d : extra) {
                bool overlaps = false;
                for (const auto& existing : out) {
                    double intersection = (d.box & existing.box).area();
                    if (intersection > .15 * std::min(d.box.area(), existing.box.area()) ||
                        cv::norm(d.centre - existing.centre) < 12) {
                        overlaps = true;
                        break;
                    }
                }
                if (!overlaps) out.push_back(d);
            }
            // Separate rescue pass preserves every previously accepted contour.
            ColourModel green = dim;
            green.greenRescue = true;
            DetectionDiagnostics greenDiag;
            auto greenExtra = detect(image, green, diagnostics ? &greenDiag : nullptr);
            if (diagnostics)
                for (auto row : greenDiag.rows) {
                    if (row.empty() || row[0] != '1') continue;
                    auto at = row.rfind(',');
                    if (at != std::string::npos) row.insert(at + 1, "green_rescue_");
                    diagnostics->rows.push_back(row);
                }
            for (auto d : greenExtra) {
                if (d.colour != 1) continue;
                bool overlaps = false;
                for (const auto& existing : out) {
                    double intersection = (d.box & existing.box).area();
                    if (intersection > .15 * std::min(d.box.area(), existing.box.area()) ||
                        cv::norm(d.centre - existing.centre) < 12) {
                        overlaps = true;
                        break;
                    }
                }
                if (!overlaps) out.push_back(d);
            }
        }
        if (cfg.coherenceKernel == 0) {
            ColourModel coherent = cfg;
            coherent.staticStyle = false;
            coherent.greenRescue = false;
            coherent.sat = 65;
            coherent.val = 25;
            coherent.minArea = std::max(100, int(cfg.minArea * .60));
            coherent.coherenceKernel = 7;
            auto fresh = detect(image, coherent);
            for (auto d : fresh)
                if (d.colour == 1) {
                    bool overlaps = false;
                    for (const auto& existing : out)
                        if ((d.box & existing.box).area() > .15 * std::min(d.box.area(), existing.box.area()) ||
                            cv::norm(d.centre - existing.centre) < 12) {
                            overlaps = true;
                            break;
                        }
                    if (!overlaps) {
                        // Opening must not turn a bridged pair or an L-shape into a
                        // new standalone block. Check its unbroken parent component.
                        auto parentModel = coherent;
                        parentModel.sat = 35;
                        parentModel.greenRescue = false;
                        Mat rawHSV;
                        cv::cvtColor(image, rawHSV, cv::COLOR_BGR2HSV);
                        Mat parentMask = colourThreshold(rawHSV, 1, parentModel);
                        cleanColourMask(parentMask, false);
                        std::vector<std::vector<cv::Point>> parents;
                        cv::findContours(parentMask, parents, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                        bool compactParent = false;
                        for (auto parent : parents)
                            if (cv::pointPolygonTest(parent, d.centre, false) >= 0) {
                                std::vector<cv::Point> hull;
                                cv::convexHull(parent, hull);
                                double hullArea = cv::contourArea(hull);
                                auto bounds = cv::boundingRect(parent);
                                compactParent = hullArea > 0 && cv::contourArea(parent) / hullArea >= .85 &&
                                    double(std::max(bounds.width, bounds.height)) /
                                    std::max(1, std::min(bounds.width, bounds.height)) <
                                    1.7;
                                break;
                            }
                        if (compactParent) out.push_back(d);
                    }
                }
        }
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) {
            if (a.centre.y != b.centre.y) return a.centre.y < b.centre.y;
            return a.centre.x < b.centre.x;
            });
        return out;
    }

    // ------------------------------------------------------------
    //  1b  Top face estimation
    //    A block away from the camera centre shows part of a side face. These functions find the square
    //    top face inside the coloured outline so the gripper aims at the real block centre.
    // ------------------------------------------------------------
    double v11SquareScore(const calibrationui::Detection& d, const calibrationui::H& h) {
        if (d.quad.size() != 4) return 1e9;
        vector<Point2d> q;
        for (auto p : d.quad) q.push_back(calibrationui::project(h, p));
        // Paint may stop short of a 35mm cube's physical edges. Validate a compact
        // square face, rather than requiring every painted edge to measure35mm.
        double meanEdge = 0;
        for (int i = 0; i < 4; ++i) meanEdge += norm(q[(i + 1) % 4] - q[i]);
        meanEdge /= 4;
        double score = 0;
        for (int i = 0; i < 4; ++i) {
            auto u = q[(i + 1) % 4] - q[i], v = q[(i + 3) % 4] - q[i];
            double length = norm(u), other = norm(v);
            if (length < 27.0 || length > 42.0 || other < 1) return 1e9;
            double cosine = abs(u.dot(v)) / (length * other);
            if (cosine > .20) return 1e9;
            score += abs(length - meanEdge) / meanEdge + cosine;
        }
        return score / 4;
    }
    calibrationui::Detection v11TopFace(const Mat& frame, const calibrationui::ColourModel& cfg,
        const calibrationui::H& h, calibrationui::Detection original,
        bool halfwayIfAmbiguous = false) {
        original.pickFaceVerified = false;
        if (frame.empty() || original.quad.size() != 4) return original;
        original.centre = calibrationui::crossCentre(original.quad);
        Rect roi = Rect(original.box.x - 12, original.box.y - 12, original.box.width + 24, original.box.height + 24) &
            Rect(0, 0, frame.cols, frame.rows);
        // The initial high-threshold contour can be a small fragment. Search a
        // bounded physical neighbourhood so its missing edge is not clipped away.
        auto worldCentre = calibrationui::project(h, original.centre);
        vector<Point2f> searchCorners;
        auto inverse = h.inv();
        for (auto offset : vector<Point2d>{ {-30, -30}, {30, -30}, {30, 30}, {-30, 30} }) {
            auto px = calibrationui::project(inverse, worldCentre + offset);
            if (isfinite(px.x) && isfinite(px.y)) searchCorners.emplace_back(float(px.x), float(px.y));
        }
        if (searchCorners.size() == 4) {
            auto expanded = boundingRect(searchCorners) & Rect(0, 0, frame.cols, frame.rows);
            if (size_t(expanded.area()) < frame.total() / 2) roi = roi | expanded;
        }
        if (roi.width < 10 || roi.height < 10) return original;
        Mat hsv, gray;
        cvtColor(frame(roi), hsv, COLOR_BGR2HSV);
        extractChannel(hsv, gray, 2);
        auto model = cfg;
        model.staticStyle = false;
        model.greenRescue = true;
        model.sat = min(model.sat, 35);
        model.val = min(model.val, 25);
        if (original.colour == 0 && model.hue[0] == 0) model.tol[0] = 25;
        Mat colour = calibrationui::colourThreshold(hsv, original.colour, model);
        vector<uchar> values;
        for (int y = 0; y < gray.rows; ++y)
            for (int x = 0; x < gray.cols; ++x)
                if (colour.at<uchar>(y, x)) values.push_back(gray.at<uchar>(y, x));
        vector<calibrationui::Detection> candidates;
        if (v11SquareScore(original, h) < .14) {
            original.pickFaceVerified = true;
            candidates.push_back(original);
        }
        if (values.size() >= 100) {
            Mat samples(1, int(values.size()), CV_8U, values.data()), ignored;
            double cut = threshold(samples, ignored, 0, 255, THRESH_BINARY | THRESH_OTSU);
            auto minmax = minmax_element(values.begin(), values.end());
            for (int side = -6; side < 2; ++side) {
                if (side == -6 && original.colour != 1) continue;
                if (side >= 0 && int(*minmax.second) - int(*minmax.first) < 22) continue;
                Mat split;
                if (side <= -3) {
                    auto coherent = model;
                    coherent.sat = side == -6 ? 45 : side == -3 ? 65 : side == -4 ? 95 : 125;
                    if (original.colour == 1 && side == -4) {
                        coherent.sat = 55;
                        coherent.hue[1] = max(0, cfg.hue[1] - 15);
                        coherent.tol[1] = 10;
                    }
                    if (original.colour == 1 && side == -5) {
                        coherent.sat = 65;
                        coherent.hue[1] = max(0, cfg.hue[1] - 5);
                        coherent.tol[1] = 15;
                    }
                    coherent.greenRescue = false;
                    split = calibrationui::colourThreshold(hsv, original.colour, coherent);
                }
                else if (side < 0)
                    split = colour.clone();
                else {
                    threshold(gray, split, cut, 255, side ? THRESH_BINARY_INV : THRESH_BINARY);
                    split &= colour;
                }
                // Close narrow paint scratches only on the unsplit colour mask; do not
                // bridge the brightness boundary that separates a visible cube side.
                if (side == -1) morphologyEx(split, split, MORPH_CLOSE, getStructuringElement(MORPH_RECT, { 5, 5 }));
                morphologyEx(split, split, MORPH_OPEN,
                    getStructuringElement(MORPH_RECT, side == -6 ? Size(11, 11)
                        : original.colour == 1 && side == -4 ? Size(9, 9)
                        : side <= -2 ? Size(7, 7)
                        : Size(3, 3)));
                if (side <= -3 && !(original.colour == 1 && side == -4) &&
                    int(*minmax.second) - int(*minmax.first) >= 22)
                    erode(split, split, getStructuringElement(MORPH_RECT, { 3, 3 }));
                vector<vector<Point>> contours;
                findContours(split, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                for (auto contour : contours) {
                    double area = abs(contourArea(contour));
                    if (area < 80) continue;
                    vector<Point> hull, quad;
                    convexHull(contour, hull);
                    double hullArea = abs(contourArea(hull));
                    if (hullArea <= 0 || area / hullArea < .85) continue;
                    bool ok = false;
                    for (double eps : {.018, .025, .035, .045, .055, .065, .075}) {
                        approxPolyDP(hull, quad, arcLength(hull, true) * eps, true);
                        if (quad.size() == 4 && isContourConvex(quad)) {
                            ok = true;
                            break;
                        }
                    }
                    if (!ok) {
                        auto rect = minAreaRect(contour);
                        double rectangleArea = rect.size.area();
                        if (rectangleArea > 0 && area / rectangleArea >= .80 && hullArea / rectangleArea >= .88) {
                            Point2f corners[4];
                            rect.points(corners);
                            quad.clear();
                            for (auto p : corners) quad.emplace_back(cvRound(p.x), cvRound(p.y));
                            ok = true;
                        }
                    }
                    if (!ok) continue;
                    auto box = boundingRect(quad);
                    if (box.x <= 1 || box.y <= 1 || box.br().x >= roi.width - 1 || box.br().y >= roi.height - 1)
                        continue;
                    calibrationui::Detection d;
                    d.colour = original.colour;
                    d.area = area;
                    for (auto& q : quad) q += roi.tl();
                    d.quad = quad;
                    d.box = boundingRect(quad);
                    d.centre = calibrationui::crossCentre(quad);
                    if (v11SquareScore(d, h) >= .14) {
                        auto rect = minAreaRect(contour);
                        double rectangleArea = rect.size.area();
                        if (rectangleArea > 0 && area / rectangleArea >= .80 && hullArea / rectangleArea >= .88) {
                            Point2f corners[4];
                            rect.points(corners);
                            d.quad.clear();
                            for (auto p : corners) d.quad.emplace_back(cvRound(p.x) + roi.x, cvRound(p.y) + roi.y);
                            d.box = boundingRect(d.quad);
                            d.centre = calibrationui::crossCentre(d.quad);
                        }
                    }
                    if (v11SquareScore(d, h) < .14 && norm(calibrationui::project(h, d.centre) - worldCentre) <= 12) {
                        d.pickFaceVerified = true;
                        candidates.push_back(d);
                    }
                }
            }
        }
        if (candidates.empty()) {
            // V13.7: no clean square at all (typically dark BLUE with LIGHT OFF, side face merged into the
            // top). Calibration still refuses. A pick uses the block outline centre with a nominal 35 mm face,
            // as long as the outline is block sized; the centre-line Y offset absorbs the side-face bias.
            if (!halfwayIfAmbiguous) return original;
            vector<Point2d> w;
            for (auto q : original.quad) w.push_back(calibrationui::project(h, Point2d(q)));
            double e0 = norm(w[1] - w[0]), e1 = norm(w[2] - w[1]), e2 = norm(w[3] - w[2]), e3 = norm(w[0] - w[3]);
            double lo = min(min(e0, e1), min(e2, e3)), hi = max(max(e0, e1), max(e2, e3));
            if (!(lo >= 28 && hi <= 55)) return original;
            Point2d mid = (w[0] + w[1] + w[2] + w[3]) * .25;
            Point2d u = (w[1] - w[0]) + (w[2] - w[3]);
            double n = norm(u);
            if (n < 1e-6) return original;
            u *= 1.0 / n;
            Point2d v(-u.y, u.x);
            auto inv = h.inv();
            original.quad.clear();
            for (auto s : vector<Point2d>{ {-1, -1}, {1, -1}, {1, 1}, {-1, 1} }) {
                auto px = calibrationui::project(inv, mid + u * (17.5 * s.x) + v * (17.5 * s.y));
                original.quad.emplace_back(cvRound(px.x), cvRound(px.y));
            }
            original.box = cv::boundingRect(original.quad);
            original.centre = calibrationui::project(inv, mid);
            original.pickFaceVerified = true;
            return original;
        }
        sort(candidates.begin(), candidates.end(),
            [&](const calibrationui::Detection& a, const calibrationui::Detection& b) {
                return v11SquareScore(a, h) < v11SquareScore(b, h);
            });
        auto best = candidates.front();
        auto xy = calibrationui::project(h, best.centre);
        double score = v11SquareScore(best, h);
        // Two squares score alike but sit >3 mm apart (top only vs top + part of the side).
        // Calibration refuses. Picks (V13) use the halfway point: it does not flip between the two from
        // frame to frame, so the position readings stop restarting. The centre-line Y offset handles the bias.
        vector<Point2d> close{ xy };
        bool ambiguous = false;
        for (size_t i = 1; i < candidates.size(); ++i)
            if (v11SquareScore(candidates[i], h) <= score + .025) {
                auto c = calibrationui::project(h, candidates[i].centre);
                if (norm(c - xy) > 3) ambiguous = true;
                close.push_back(c);
            }
        if (!ambiguous) return best;
        if (!halfwayIfAmbiguous) {
            original.pickFaceVerified = false;
            return original;
        }
        Point2d mean(0, 0);
        for (auto c : close) mean += c;
        mean *= 1.0 / close.size();
        Point2d shift = mean - xy;
        auto inv = h.inv();
        for (auto& q : best.quad) {
            auto w = calibrationui::project(h, Point2d(q)) + shift;
            auto px = calibrationui::project(inv, w);
            q = cv::Point(cvRound(px.x), cvRound(px.y));
        }
        best.box = cv::boundingRect(best.quad);
        best.centre = calibrationui::project(inv, mean);
        best.pickFaceVerified = true;
        return best;
    }
    // ---- Centre-line top face (disabled, kept for reference)
    // Directly below the camera no side face is visible. Further away, the side that faces the camera's
    // centre line shows in the image and grows with the distance (perspective). The top face's OUTER
    // edge (away from the centre line) stays crisp against the background in any colour or lighting, so
    // where the outline is longer than a 35 mm face, the centre is placed 17.5 mm in from that outer edge.
    // How much side is visible depends on colour (19 Oct frames: blue 5.7, red 3.0, green 0.5 mm), which
    // is why the amount is measured per block and only the DIRECTION comes from the centre line.
    // OFF by default: the 19 Oct fit placed the centre line at Y 28 from side SIZES alone, but measured side
    // DIRECTIONS show +Y sides even at Y +65..+74, so that fit moved those targets into the side. Enable only
    // with a MEASURED centre line (block under the camera with no visible side, XY read in Test).
    struct V13Optics {
        bool enabled = false;
        double nadirX = 510, nadirY = 28, sidePerMm = 0.04;
    };
    inline const V13Optics& v13Optics() {
        static V13Optics o;
        static bool loaded = false;
        if (!loaded) {
            loaded = true;
            bool exists = std::ifstream("v13_camera_centre.yml").good();
            cv::FileStorage f;
            if (exists) f.open("v13_camera_centre.yml", cv::FileStorage::READ);
            if (exists && f.isOpened()) {
                int on = 0;
                if (!f["enabled"].empty()) f["enabled"] >> on;
                o.enabled = on == 1;
                if (!f["nadir_x_mm"].empty()) f["nadir_x_mm"] >> o.nadirX;
                if (!f["nadir_y_mm"].empty()) f["nadir_y_mm"] >> o.nadirY;
                if (!f["side_mm_per_mm"].empty()) f["side_mm_per_mm"] >> o.sidePerMm;
            }
            else {
                cv::FileStorage w("v13_camera_centre.yml", cv::FileStorage::WRITE);
                if (w.isOpened())
                    w << "note"
                    << "Set enabled 1 only after MEASURING the robot XY directly under the camera (a block there shows no side face)"
                    << "enabled" << 0 << "nadir_x_mm" << o.nadirX << "nadir_y_mm" << o.nadirY << "side_mm_per_mm"
                    << o.sidePerMm;
            }
            if (!std::isfinite(o.nadirX) || !std::isfinite(o.nadirY) || !std::isfinite(o.sidePerMm) ||
                o.sidePerMm < 0 || o.sidePerMm > .2)
                o = V13Optics{};
            if (o.enabled)
                std::cout << "Centre-line correction ON: nadir X " << o.nadirX << " Y " << o.nadirY << " mm"
                << std::endl;
        }
        return o;
    }
    // Whole same-colour outline of one block (top + any visible side) in robot mm. The saturation floor
    // follows the local background, so a lamp-tinted background (LIGHT OFF) is not counted as block.
    inline bool v13Outline(const cv::Mat& frame, const ColourModel& cfg, const H& h, const Detection& o, double& x0,
        double& x1, double& y0, double& y1) {
        if (frame.empty() || o.quad.size() != 4) return false;
        auto wc = project(h, crossCentre(o.quad));
        auto inv = h.inv();
        std::vector<cv::Point2f> s;
        for (auto off : std::vector<Point2d>{ {-40, -40}, {40, -40}, {40, 40}, {-40, 40} }) {
            auto p = project(inv, wc + off);
            if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
            s.emplace_back(float(p.x), float(p.y));
        }
        cv::Rect roi = (cv::boundingRect(s) | o.box) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.width < 10 || roi.height < 10) return false;
        cv::Mat hsv;
        cv::cvtColor(frame(roi), hsv, cv::COLOR_BGR2HSV);
        std::vector<uchar> sat;
        for (int y = 0; y < hsv.rows; y += 2)
            for (int x = 0; x < hsv.cols; x += 2) sat.push_back(hsv.at<cv::Vec3b>(y, x)[1]);
        std::nth_element(sat.begin(), sat.begin() + sat.size() / 2, sat.end());
        int background = sat[sat.size() / 2];
        auto m = cfg;
        m.staticStyle = false;
        m.greenRescue = false;
        m.sat = std::max(45, std::min(150, background + 35));
        m.val = std::min(m.val, 25);
        const int df[3] = { 0, 60, 110 }, tl[3] = { 13, 25, 20 };
        for (int k = 0; k < 3; ++k)
            if (m.hue[k] == df[k]) m.tol[k] = std::max(m.tol[k], tl[k]);
        cv::Mat mask = colourThreshold(hsv, o.colour, m);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, { 3, 3 }));
        cv::Mat lab, st, ce;
        cv::connectedComponentsWithStats(mask, lab, st, ce, 8, CV_32S);
        cv::Point seed(cvRound(o.centre.x) - roi.x, cvRound(o.centre.y) - roi.y);
        if (seed.x < 0 || seed.y < 0 || seed.x >= lab.cols || seed.y >= lab.rows) return false;
        int id = lab.at<int>(seed);
        if (id <= 0) return false;
        std::vector<double> X, Y;
        for (int r = 0; r < lab.rows; ++r)
            for (int c = 0; c < lab.cols; ++c)
                if (lab.at<int>(r, c) == id) {
                    auto w = project(h, Point2d(c + roi.x, r + roi.y));
                    X.push_back(w.x);
                    Y.push_back(w.y);
                }
        if (X.size() < 100) return false;
        std::sort(X.begin(), X.end());
        std::sort(Y.begin(), Y.end());
        x0 = X[X.size() * 2 / 100];
        x1 = X[X.size() * 98 / 100];
        y0 = Y[Y.size() * 2 / 100];
        y1 = Y[Y.size() * 98 / 100];
        return true;
    }
    inline double v13AnchorAxis(double lo, double hi, double nadir, double k) {
        double mid = (lo + hi) * .5, excess = hi - lo - 35, predicted = k * std::abs(mid - nadir);
        if (excess < 1.5 || predicted < 1) return mid; // no visible side on this axis
        if (excess > 3 + 2.5 * predicted)
            return std::numeric_limits<double>::quiet_NaN(); // more than a side: do not guess
        return mid > nadir ? hi - 17.5 : lo + 17.5;          // side lies towards the centre line
    }
    // Shared by calibration and pickup so the mapping and the pick targets use the same centre.
    inline Detection v13PickFace(const cv::Mat& frame, const ColourModel& cfg, const H& h, const Detection& original,
        bool forPick = false) {
        Detection d = v11TopFace(frame, cfg, h, original, forPick);
        if (!d.pickFaceVerified || !v13Optics().enabled) return d;
        double x0, x1, y0, y1;
        if (!v13Outline(frame, cfg, h, d, x0, x1, y0, y1)) return d;
        const auto& o = v13Optics();
        double cx = v13AnchorAxis(x0, x1, o.nadirX, o.sidePerMm), cy = v13AnchorAxis(y0, y1, o.nadirY, o.sidePerMm);
        if (!std::isfinite(cx) || !std::isfinite(cy)) return d; // keep the V12 face; the Y guard still applies
        auto centre = project(h, d.centre);
        Point2d shift(cx - centre.x, cy - centre.y);
        if (norm(shift) < .05) return d;
        auto inv = h.inv();
        for (auto& q : d.quad) {
            auto w = project(h, Point2d(q)) + shift;
            auto px = project(inv, w);
            q = cv::Point(cvRound(px.x), cvRound(px.y));
        }
        d.box = cv::boundingRect(d.quad);
        d.centre = project(inv, centre + shift);
        return d;
    }
    // Track only fresh detections; retained association history never creates an output detection.
    struct FreshTrack {
        int colour = 0, missed = 0;
        Point2d last, anchor;
        Detection observation;
        std::deque<Point2d> samples;
    };
    struct DetectionFilter {
        std::vector<FreshTrack> tracks;
        std::vector<Detection> reacquire(const cv::Mat& image, const ColourModel& cfg,
            std::vector<Detection> fresh) const {
            // Retry only near recently observed geometry. Every output comes from this image.
            ColourModel local = cfg;
            local.staticStyle = false;
            local.greenRescue = true;
            local.sat = std::min(local.sat, 35);
            local.val = std::min(local.val, 25);
            const int defaults[3] = { 0, 60, 110 }, tol[3] = { 13, 25, 20 };
            for (int k = 0; k < 3; ++k)
                if (local.hue[k] == defaults[k]) local.tol[k] = std::max(local.tol[k], tol[k]);
            for (const auto& t : tracks) {
                if (t.missed > 2 || t.observation.box.area() == 0) continue;
                bool present = false;
                for (const auto& d : fresh)
                    if (cv::norm(d.centre - t.last) < 12 ||
                        (d.box & t.observation.box).area() > .15 * std::min(d.box.area(), t.observation.box.area()))
                        present = true;
                if (present) continue;
                auto old = t.observation.box;
                int pad = std::max(10, int(std::sqrt(old.area()) * .35));
                cv::Rect roi = cv::Rect(old.x - pad, old.y - pad, old.width + 2 * pad, old.height + 2 * pad) &
                    cv::Rect(0, 0, image.cols, image.rows);
                local.minArea = std::max(80, int(t.observation.area * .60));
                auto candidates = detect(image(roi), local);
                std::vector<Detection> matches;
                for (auto d : candidates) {
                    if (d.colour != t.colour || d.area < t.observation.area * .60 || d.area > t.observation.area * 1.60)
                        continue;
                    d.centre += Point2d(roi.x, roi.y);
                    d.box += roi.tl();
                    for (auto& q : d.quad) q += roi.tl();
                    if (cv::norm(d.centre - t.last) <= 8) matches.push_back(d);
                }
                if (matches.size() == 1) fresh.push_back(matches.front());
            }
            std::sort(fresh.begin(), fresh.end(), [](const Detection& a, const Detection& b) {
                return a.centre.y == b.centre.y ? a.centre.x < b.centre.x : a.centre.y < b.centre.y;
                });
            return fresh;
        }
        std::vector<Detection> update(std::vector<Detection> raw) {
            for (auto& t : tracks) ++t.missed;
            for (auto& d : raw) {
                int match = -1;
                double best = 12;
                for (size_t i = 0; i < tracks.size(); ++i) {
                    auto& t = tracks[i];
                    double gap = cv::norm(d.centre - t.last);
                    if (t.missed > 0 && t.missed <= 4 && t.colour == d.colour && gap < best) {
                        best = gap;
                        match = int(i);
                    }
                }
                if (match < 0) {
                    FreshTrack t;
                    t.colour = d.colour;
                    t.last = t.anchor = d.centre;
                    tracks.push_back(t);
                    match = int(tracks.size() - 1);
                }
                auto& t = tracks[match];
                if (t.missed > 3 || cv::norm(d.centre - t.anchor) > 2.5) {
                    t.samples.clear();
                    t.anchor = d.centre;
                }
                t.observation = d;
                t.last = d.centre;
                t.missed = 0;
                t.samples.push_back(d.centre);
                if (t.samples.size() > 5) t.samples.pop_front();
                std::vector<double> x, y;
                for (auto q : t.samples) {
                    x.push_back(q.x);
                    y.push_back(q.y);
                }
                std::sort(x.begin(), x.end());
                std::sort(y.begin(), y.end());
                d.centre = { x[x.size() / 2], y[y.size() / 2] };
            }
            tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const FreshTrack& t) { return t.missed > 3; }),
                tracks.end());
            return raw;
        }
        void clear() { tracks.clear(); }
    };
    struct SelectedStability {
        std::deque<Point2d> samples;
        Point2d centre{};
        int good = 0, missed = 0;
        bool ready = false;
        double drift = 0;
        std::string reason = "select a block";
        static Point2d median(const std::deque<Point2d>& p, size_t first, size_t count) {
            std::vector<double> x, y;
            for (size_t i = first; i < first + count; ++i) {
                x.push_back(p[i].x);
                y.push_back(p[i].y);
            }
            std::sort(x.begin(), x.end());
            std::sort(y.begin(), y.end());
            return { (x[(x.size() - 1) / 2] + x[x.size() / 2]) * .5, (y[(y.size() - 1) / 2] + y[y.size() / 2]) * .5 };
        }
        static Point2d clusterCentre(const std::deque<Point2d>& p) {
            std::vector<double> x, y;
            for (auto q : p) {
                x.push_back(q.x);
                y.push_back(q.y);
            }
            std::sort(x.begin(), x.end());
            std::sort(y.begin(), y.end());
            size_t trim = p.size() / 6;
            Point2d c{};
            for (size_t i = trim; i < p.size() - trim; ++i) c += Point2d(x[i], y[i]);
            return c / double(p.size() - 2 * trim);
        }
        void clear() {
            samples.clear();
            good = 0;
            missed = 0;
            ready = false;
            drift = 0;
            reason = "collecting fresh selected-block samples";
        }
        void update(Point2d p, bool fresh) {
            ready = false;
            if (!fresh) {
                ++missed;
                good = 0;
                reason = "selected block missing in current frame";
                if (missed > 2) samples.clear();
                return;
            }
            if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
                clear();
                reason = "invalid selected centre";
                return;
            }
            if (missed > 2) samples.clear();
            missed = 0;
            if (!samples.empty() && cv::norm(p - median(samples, 0, samples.size())) > 6) {
                clear();
                reason = "block moved: collecting at new position";
            }
            samples.push_back(p);
            if (samples.size() > 12) samples.pop_front();
            centre = clusterCentre(samples);
            good = 0;
            for (auto q : samples)
                if (cv::norm(q - centre) <= 2.5) ++good;
            if (samples.size() < 8) {
                reason = "collecting " + std::to_string(samples.size()) + "/8 fresh samples";
                return;
            }
            drift = cv::norm(median(samples, 0, 4) - median(samples, samples.size() - 4, 4));
            if (cv::norm(p - centre) > 2.5) {
                reason = "current centre outside 2.5px stationary cluster";
                return;
            }
            if (drift > 1.5) {
                reason = "centre drifting " + std::to_string(drift).substr(0, 4) + "px (limit 1.5)";
                return;
            }
            if (good < 8) {
                reason = "noisy centre: " + std::to_string(good) + "/8 clustered samples";
                return;
            }
            ready = true;
            reason = "ready: fresh robust centre locked on Send";
        }
    };

    // ------------------------------------------------------------
    //  1c  Camera to robot mapping
    //    Orders the nine detected calibration blocks as P1 to P9 and fits the homography (pixels -> mm).
    // ------------------------------------------------------------
    bool gridOrder(std::vector<Detection>& d, bool flipRows, bool flipCols) {
        if (d.size() != 9) return false;
        std::sort(d.begin(), d.end(), [](auto& a, auto& b) { return a.centre.y < b.centre.y; });
        double meanSize = 0;
        for (auto& a : d) meanSize += std::sqrt(a.area);
        meanSize /= 9;
        for (int r = 0; r < 3; ++r) {
            std::sort(d.begin() + r * 3, d.begin() + r * 3 + 3,
                [](auto& a, auto& b) { return a.centre.x < b.centre.x; });
            if (d[r * 3 + 2].centre.x - d[r * 3].centre.x < meanSize) return false;
            if (r && d[r * 3].centre.y - d[(r - 1) * 3].centre.y < meanSize * .5) return false;
        }
        for (int i = 0; i < 9; ++i)
            for (int j = i + 1; j < 9; ++j)
                if (cv::norm(d[i].centre - d[j].centre) < meanSize * .7) return false;
        auto ordered = d;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) d[r * 3 + c] = ordered[(flipRows ? 2 - r : r) * 3 + (flipCols ? 2 - c : c)];
        return true;
    }
    struct Mapping {
        H h = H::eye();
        std::vector<Point2d> pixels, robot;
        cv::Size resolution;
        double rms = 0, max = 0;
    };
    // Fits the pixel -> robot mm homography from the nine ordered calibration blocks.
    bool makeMapping(const std::vector<Detection>& d, const std::vector<Point2d>& robot, cv::Size resolution,
        Mapping& m) {
        if (d.size() != 9) return false;
        std::vector<Point2d> p;
        for (auto& a : d) p.push_back(a.centre);
        if (!fitH(p, robot, m.h)) return false;
        m.pixels = p;
        m.robot = robot;
        m.resolution = resolution;
        m.rms = 0;
        m.max = 0;
        for (int i = 0; i < 9; ++i) {
            double e = cv::norm(project(m.h, p[i]) - robot[i]);
            m.rms += e * e;
            m.max = std::max(m.max, e);
        }
        m.rms = std::sqrt(m.rms / 9);
        return true;
    }
    double regionDistance(const Mapping& m, Point2d robot) {
        std::vector<cv::Point2f> hull;
        for (int i : {0, 2, 8, 6}) hull.emplace_back(float(m.robot[i].x), float(m.robot[i].y));
        return cv::pointPolygonTest(hull, { float(robot.x), float(robot.y) }, true);
    }
    bool inRegion(const Mapping& m, Point2d robot) {
        return regionDistance(m, robot) >= 0;
    }
    bool nearBoundary(const Mapping& m, Point2d robot) {
        return std::abs(regionDistance(m, robot)) <= 2.0;
    } // display only; does not expand planning region
    std::string regionLabel(const Mapping& m, Point2d p) {
        return nearBoundary(m, p) ? "EDGE (within 2mm)" : inRegion(m, p) ? "INSIDE" : "OUTSIDE";
    }
    // Whole-table test area: any block the camera sees is allowed if the ROBOT can reach it.
    // (Outside the nine calibrated blocks the camera mapping is extrapolated, so accuracy drops with distance from the grid.)
    constexpr double REACH_MIN_MM = 360.0, REACH_MAX_MM = 540.0,
        REACH_X_MIN_MM = 200.0; // distance from robot base, X limit
    inline bool reachOk(Point2d p) {
        double r = std::hypot(p.x, p.y);
        return std::isfinite(r) && p.x >= REACH_X_MIN_MM && r >= REACH_MIN_MM && r <= REACH_MAX_MM;
    }
    struct Options {
        std::string source = "camera", image, reference = "reference_points.csv";
        int camera = 0;
        double fitLimit = 2;
        bool explicitReference = false, selftest = false, snapshot = false;
    };
    struct Button {
        cv::Rect r;
        std::string label;
        char action;
    };
    struct Plan {
        bool valid = false;
        std::array<int, 3> source{ -1, -1, -1 };
        std::array<Point2d, 3> xy{};
        std::array<int, 3> colours{};
        std::string reason;
    };
    // Same top-to-bottom patterns as the original planning main.cpp; detected colours use 0-based IDs here.
    const std::array<std::array<int, 3>, 3> PATTERNS{ {{{1, 0, 2}}, {{0, 1, 2}}, {{2, 1, 0}}} };
    const char* COLOUR_NAMES[] = { "RED", "GREEN", "BLUE" };
    Plan makePlan(const std::vector<Detection>& detections, const Mapping& mapping, int routine) {
        Plan p;
        if (routine < 1 || routine > 3) {
            p.reason = "Invalid pattern";
            return p;
        }
        for (int l = 0; l < 3; ++l) {
            int wanted = PATTERNS[routine - 1][2 - l];
            p.colours[l] = wanted;
            for (int j = 0; j < int(detections.size()); ++j) {
                if (detections[j].colour != wanted) continue;
                bool used = false;
                for (int k = 0; k < l; ++k)
                    if (p.source[k] == j) used = true;
                if (used) continue;
                Point2d xy = project(mapping.h, detections[j].centre);
                if (!inRegion(mapping, xy)) continue;
                p.source[l] = j;
                p.xy[l] = xy;
                break;
            }
            if (p.source[l] < 0) {
                p.reason = std::string("Need in-region ") + COLOUR_NAMES[wanted];
                return p;
            }
        }
        p.valid = true;
        p.reason = "Preview ready; not a motion approval";
        return p;
    }

    // ------------------------------------------------------------
    //  1d  Calibration page state and drawing
    // ------------------------------------------------------------
    struct App {
        Options opt;
        ColourModel cfg, staticCfg;
        bool useStaticDetector = false, staticColourEdited = false;
        const ColourModel& activeColourModel() const { return test && useStaticDetector ? staticCfg : cfg; }
        cv::Mat frame, ui;
        std::vector<Detection> detections, grid, previous;
        std::vector<Point2d> refs = nominalRobot();
        Mapping mapping;
        bool v11TopFaceCalibration = true; // loaded legacy files explicitly clear this marker
        bool test = false, valid = false, labelsOK = false, refsOK = false, planeOK = false, flipRows = false,
            flipCols = false, freeze = false, missing = false, noise = false, exit = false, selectStable = false;
        struct CompareRow {
            Point2d reference, predicted;
            double error;
        };
        std::vector<CompareRow> comparison;
        bool comparisonView = false;
        std::string comparisonPath;
        bool enlargedValidation = false, enlargedApproved = false;
        H enlargedMapping = H::eye();
        cv::Size enlargedResolution;
        std::vector<Point2d> enlargedPredicted, enlargedActual;
        std::vector<cv::Point2f> enlargedHull;
        bool enlargedMatches() const {
            return valid && opt.source == "camera" && enlargedResolution == frame.size() &&
                cv::norm(cv::Mat(enlargedMapping) - cv::Mat(mapping.h)) < 1e-8;
        }
        bool enlargedCoverage(Point2d p) const {
            return enlargedApproved && enlargedMatches() && enlargedHull.size() >= 3 &&
                cv::pointPolygonTest(enlargedHull, p, true) >= 2;
        }
        bool enlargedQuality(std::string& why) {
            if (!enlargedMatches()) {
                why = "Mapping/resolution changed; collect fresh checks";
                return false;
            }
            if (enlargedPredicted.size() < 9 || enlargedActual.size() != enlargedPredicted.size()) {
                why = "Need at least 9 distinct measured checks";
                return false;
            }
            for (size_t i = 0; i < enlargedActual.size(); ++i) {
                if (!std::isfinite(enlargedActual[i].x) || !std::isfinite(enlargedActual[i].y) ||
                    !std::isfinite(enlargedPredicted[i].x) || !std::isfinite(enlargedPredicted[i].y) ||
                    cv::norm(enlargedActual[i] - enlargedPredicted[i]) > 2) {
                    why = "A measured XY error exceeds 2.00mm; correct mapping and recheck";
                    return false;
                }
                for (size_t j = 0; j < i; ++j)
                    if (cv::norm(enlargedPredicted[i] - enlargedPredicted[j]) < 25) {
                        why = "Checks must be at least 25mm apart";
                        return false;
                    }
            }
            std::vector<cv::Point2f> pts;
            for (auto p : enlargedPredicted) pts.emplace_back(float(p.x), float(p.y));
            cv::convexHull(pts, enlargedHull);
            if (std::abs(cv::contourArea(enlargedHull)) <= polygonArea(mapping.robot)) {
                why = "Check corners/edges of an area LARGER than current calibration";
                return false;
            }
            bool interior = false;
            for (auto p : pts)
                if (cv::pointPolygonTest(enlargedHull, p, true) >= 20) interior = true;
            if (!interior) {
                why = "Include an independent interior check, at least 20mm inside outer checks";
                return false;
            }
            why = "Measured XY envelope accepted; physical workspace/reach checks still required";
            return true;
        }
        void saveEnlarged() {
            cv::FileStorage f("enlarged_validation.yml", cv::FileStorage::WRITE);
            if (!f.isOpened()) throw std::runtime_error("Cannot save enlarged validation");
            f << "mapping" << cv::Mat(enlargedMapping) << "width" << enlargedResolution.width << "height"
                << enlargedResolution.height << "approved" << int(enlargedApproved) << "predicted" << enlargedPredicted
                << "actual" << enlargedActual;
        }
        void loadEnlarged() {
            if (!fileutil::exists("enlarged_validation.yml")) return;
            cv::FileStorage f("enlarged_validation.yml", cv::FileStorage::READ);
            cv::Mat h;
            int w = 0, ht = 0, approved = 0;
            f["mapping"] >> h;
            f["width"] >> w;
            f["height"] >> ht;
            f["approved"] >> approved;
            if (h.rows != 3 || h.cols != 3 || h.type() != CV_64F) return;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) enlargedMapping(r, c) = h.at<double>(r, c);
            enlargedResolution = { w, ht };
            f["predicted"] >> enlargedPredicted;
            f["actual"] >> enlargedActual;
            std::string why;
            enlargedApproved = approved && enlargedQuality(why);
        }
        void resetEnlarged() {
            enlargedApproved = false;
            enlargedPredicted.clear();
            enlargedActual.clear();
            enlargedHull.clear();
            enlargedMapping = mapping.h;
            enlargedResolution = frame.size();
            saveEnlarged();
            message = "Checks cleared. Measure 4 corners, 4 edges and centre across enlarged area.";
        }
        bool enlargedHoverView = false;
        LightingInfo lighting;
        int diagnosticRemaining = 0;
        std::string diagnosticDir;
        DetectionFilter detectionFilter;
        SelectedStability selectedStability;
        bool selectionTracked = false, selectionMissing = false;
        Point2d selectionPosition;
        int selectionColour = 0;
        bool rawView = false;
        double rawPointStep = 0, rawPointDrift = 0;
        std::deque<std::vector<Point2d>> pointHistory;
        std::vector<Point2d> lastRawCentres;
        double stableSpread = 0, windowDrift = 0, noiseLimitPx = 4.0;
        bool gridQualityReady = false;
        bool fitReview = false;
        double lastFitMax = 0;
        std::array<Point2d, 9> fitDelta, fitPixels;
        std::array<double, 9> fitResidual{};
        std::string stabilityReason = "Need nine fresh detections";
        static double median(std::vector<double> v) {
            std::sort(v.begin(), v.end());
            size_t n = v.size();
            return (v[(n - 1) / 2] + v[n / 2]) * .5;
        }
        std::vector<Point2d> averagedCentres() const {
            if (pointHistory.empty()) return {};
            std::vector<Point2d> out(9);
            for (int i = 0; i < 9; ++i) {
                std::vector<double> x, y;
                for (const auto& f : pointHistory) {
                    x.push_back(f[i].x);
                    y.push_back(f[i].y);
                }
                out[i] = { median(x), median(y) };
            }
            return out;
        }
        void updateGridStability() {
            gridQualityReady = false;
            rawPointStep = 0;
            rawPointDrift = 0;
            stableSpread = 0;
            windowDrift = 0;
            if (!gridOrder(grid, flipRows, flipCols)) {
                grid.clear();
                ++missingGridFrames;
                if (!pointHistory.empty() && missingGridFrames <= 5 && lastGoodGrid.time_since_epoch().count() &&
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - lastGoodGrid).count() <= 1.5) {
                    stable = int(pointHistory.size());
                    stabilityReason = "Paused at " + std::to_string(stable) +
                        "/20: reacquiring missing top; Fit needs nine CURRENT tops";
                    return;
                }
                stable = 0;
                previous.clear();
                pointHistory.clear();
                lastRawCentres.clear();
                stabilityReason = "Need nine fresh separate tops";
                return;
            }
            if (lastGoodGrid.time_since_epoch().count() &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - lastGoodGrid).count() > 1.5) {
                previous.clear();
                pointHistory.clear();
            }
            missingGridFrames = 0;
            lastGoodGrid = std::chrono::steady_clock::now();
            bool same = previous.size() == 9;
            std::vector<Point2d> raw;
            for (int i = 0; i < 9; ++i) {
                raw.push_back(grid[i].centre);
                if (lastRawCentres.size() == 9)
                    rawPointStep = std::max(rawPointStep, cv::norm(raw.back() - lastRawCentres[i]));
                if (previous.size() == 9) {
                    double drift = cv::norm(previous[i].centre - raw.back());
                    rawPointDrift = std::max(rawPointDrift, drift);
                    if (drift > RESET_JUMP_PX || previous[i].colour != grid[i].colour) same = false;
                }
            }
            if (!same) {
                previous = grid;
                pointHistory.clear();
            }
            pointHistory.push_back(raw);
            if (pointHistory.size() > STABLE_FRAMES) pointHistory.pop_front();
            lastRawCentres = raw;
            stable = int(pointHistory.size());
            stabilityReason = "Collecting fresh 20-frame window";
            if (pointHistory.size() < STABLE_FRAMES) return;
            auto centre = averagedCentres();
            bool acceptable = true;
            for (int i = 0; i < 9; ++i) {
                std::vector<double> dist, firstX, firstY, lastX, lastY;
                for (size_t j = 0; j < pointHistory.size(); ++j) {
                    auto p = pointHistory[j][i];
                    dist.push_back(cv::norm(p - centre[i]));
                    if (j < 5) {
                        firstX.push_back(p.x);
                        firstY.push_back(p.y);
                    }
                    if (j >= 15) {
                        lastX.push_back(p.x);
                        lastY.push_back(p.y);
                    }
                }
                std::sort(dist.begin(), dist.end());
                double spread = dist[17]; // at least 18/20 samples within 2.5px
                auto mean = [](const std::vector<double>& v) {
                    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
                    };
                double drift = cv::norm(Point2d(mean(firstX), mean(firstY)) - Point2d(mean(lastX), mean(lastY)));
                stableSpread = std::max(stableSpread, spread);
                windowDrift = std::max(windowDrift, drift);
                if (spread > noiseLimitPx || drift > DRIFT_LIMIT_PX) acceptable = false;
            }
            gridQualityReady = acceptable;
            if (!acceptable) {
                stabilityReason = windowDrift > DRIFT_LIMIT_PX ? "Window drifting - keep blocks/camera still"
                    : "Point noise too high - sample colours/check light";
            }
            else
                stabilityReason = "20-frame window stable (median centres)";
        }
        int missingGridFrames = 0;
        std::chrono::steady_clock::time_point lastGoodGrid{};
        int stable = 0, selected = 0, pose = 0, trial = 0, sample = -1, snapshotNo = 0, testStable = 0;
        Point2d testPrevious{}, block{ 105, 125 };
        bool entering = false, planning = false, planLocked = false;
        int pattern = 1, planStable = 0;
        Plan lockedPlan;
        std::vector<Detection> planAnchor;
        std::string input;
        Point2d measuredPixel{}, measuredPrediction{};
        cv::Mat validationFrame;
        std::string session;
        std::string message = "Place nine cubes. Confirm P1-P9 direction and measured robot reference coordinates.";
        std::vector<Button> buttons = {
            {{15, 50, 165, 34}, "1 Calibration", '1'}, {{190, 50, 180, 34}, "2 Coordinate test", '2'},
            {{380, 50, 160, 34}, "3 Planning", '3'},   {{550, 50, 130, 34}, "C Fit + save", 'c'},
            {{690, 50, 130, 34}, "I Invalidate", 'i'}, {{830, 50, 130, 34}, "S Snapshot", 's'},
            {{970, 50, 170, 34}, "E Actual XY", 'e'},  {{1150, 50, 110, 34}, "Q Quit", 'q'} };
        std::string sourceName() const {
            return opt.source == "camera" ? "LIVE CAMERA " + std::to_string(opt.camera)
                : opt.source == "demo" ? "SYNTHETIC DEMO"
                : "IMAGE FILE";
        }
        void invalidate(const std::string& why) {
            // An explicitly invalidated camera fit must not silently return after restarting.
            if (opt.source == "camera" && fileutil::exists("calibration_saved.yml")) {
                try {
                    cv::FileStorage state("calibration_validity.yml", cv::FileStorage::WRITE);
                    if (!state.isOpened()) throw std::runtime_error("Cannot persist calibration invalidation");
                    state << "valid" << 0;
                }
                catch (const std::exception& e) {
                    std::cerr << e.what() << std::endl;
                }
            }
            fitReview = false;
            missingGridFrames = 0;
            lastGoodGrid = {};
            detectionFilter.clear();
            selectedStability.clear();
            selectStable = false;
            testStable = 0;
            selectionTracked = false;
            selectionMissing = false;
            enlargedApproved = false;
            enlargedPredicted.clear();
            enlargedActual.clear();
            enlargedHull.clear();
            valid = false;
            labelsOK = refsOK = planeOK = false;
            stable = 0;
            comparisonView = false;
            gridQualityReady = false;
            previous.clear();
            pointHistory.clear();
            lastRawCentres.clear();
            entering = false;
            planLocked = false;
            planStable = 0;
            planAnchor.clear();
            message = why;
        }
        std::string sessionDir() {
            if (session.empty()) {
                auto id = std::chrono::system_clock::now().time_since_epoch().count();
                fileutil::createDirectory("sessions");
                session = "sessions/" + (stamp() + "_" + opt.source + "_" + std::to_string(id));
                fileutil::createDirectory(session);
            }
            return session;
        }
        void update() {
            if (!test)
                cfg.staticStyle = true; // Shared segmentation with Dynamic pickup, including dim-paint fallbacks.
            if (valid && frame.size() != mapping.resolution) invalidate("Resolution changed - recalibrate.");
            detections = detectionFilter.reacquire(frame, activeColourModel(),
                detect(frame, activeColourModel(), nullptr, &lighting));
            if (!test && detections.size() == 9) {
                // Seed with the nine known references, then collect the SAME top-face centres used by pickup.
                Mapping provisional;
                if (gridOrder(detections, flipRows, flipCols) &&
                    makeMapping(detections, refs, frame.size(), provisional)) {
                    vector<Point2d> pixels;
                    for (auto d : detections) pixels.push_back(d.centre);
                    fitGridSeed(pixels, refs, provisional.h);
                    for (auto& d : detections) {
                        try {
                            d = v13PickFace(frame, activeColourModel(), provisional.h, d);
                        }
                        catch (...) {
                            d.pickFaceVerified = false;
                        }
                    }
                }
            }
            if (diagnosticRemaining > 0) {
                try {
                    captureDetectionFrame();
                }
                catch (const std::exception& e) {
                    diagnosticRemaining = 0;
                    message = std::string("DETECTION CAPTURE FAILED: ") + e.what();
                }
            }
            // Calibration retains raw centres for its existing noise/drift gates.
            auto filtered = detectionFilter.update(detections);
            if (test) detections = filtered;
            grid = detections;
            if (test && selectionTracked) {
                int match = -1;
                double best = 12;
                for (size_t i = 0; i < detections.size(); ++i) {
                    double gap = cv::norm(detections[i].centre - selectionPosition);
                    if (detections[i].colour == selectionColour && gap < best) {
                        match = int(i);
                        best = gap;
                    }
                }
                selectionMissing = match < 0;
                if (match >= 0) {
                    selected = match;
                    selectionPosition = detections[match].centre;
                }
            }
            bool unchanged = planAnchor.size() == detections.size() && !detections.empty();
            if (unchanged)
                for (size_t i = 0; i < detections.size(); ++i)
                    if (planAnchor[i].colour != detections[i].colour ||
                        cv::norm(planAnchor[i].centre - detections[i].centre) > 2.5)
                        unchanged = false;
            if (unchanged)
                planStable = std::min(1000, planStable + 1);
            else {
                planStable = 1;
                planAnchor = detections;
                if (planLocked) {
                    planLocked = false;
                    message = "Scene changed - preview unlocked. Wait for stable detections.";
                }
            }
            updateGridStability();
            if (selected >= int(detections.size())) selected = 0;
            if (test && !selectionTracked && !detections.empty()) {
                selectionTracked = true;
                selectionPosition = detections[selected].centre;
                selectionColour = detections[selected].colour;
                selectedStability.clear();
            }
            selectedStability.update(!detections.empty() && !selectionMissing ? detections[selected].centre : Point2d(),
                !detections.empty() && !selectionMissing);
            testStable = selectedStability.good;
            selectStable = selectedStability.ready;
        }
        Plan currentPlan() const {
            if (!valid) {
                Plan p;
                p.reason = "Calibrate first";
                return p;
            }
            if (planStable < STABLE_FRAMES) {
                Plan p;
                p.reason = "Wait for stable block positions";
                return p;
            }
            return planLocked ? lockedPlan : makePlan(detections, mapping, pattern);
        }
        void savePlan() {
            Plan p = currentPlan();
            if (!planning || !p.valid) {
                message = "Open Planning and wait for a valid stable plan.";
                return;
            }
            auto dir = sessionDir();
            std::string name = "planning_preview_" + std::to_string(++snapshotNo) + ".csv";
            std::ofstream f(dir + "/" + name);
            if (!f) throw std::runtime_error("Cannot save planning preview");
            f << "source,stack,level_bottom_first,colour,detection_id,pick_robot_x_mm,pick_robot_y_mm,status\n"
                << std::setprecision(12);
            for (int l = 0; l < 3; ++l)
                f << opt.source << ",S" << pattern << ',' << l + 1 << ',' << COLOUR_NAMES[p.colours[l]] << ",B"
                << p.source[l] + 1 << ',' << p.xy[l].x << ',' << p.xy[l].y << ",PREVIEW_ONLY_NO_MOTION\n";
            message = "Saved " + name + ". No commands sent; fixed S" + std::to_string(pattern) +
                " remains on the robot side.";
        }
        void calibration() {
            if (test) {
                message = "Switch to Calibration first.";
                return;
            }
            if (grid.size() != 9) {
                message = "BLOCKED: detected " + std::to_string(detections.size()) + " tops; need nine separate tops";
                return;
            }
            std::string rejected;
            for (size_t i = 0; i < grid.size(); ++i)
                if (!grid[i].pickFaceVerified) {
                    if (!rejected.empty()) rejected += ", ";
                    rejected += "P" + std::to_string(i + 1);
                }
            if (!rejected.empty()) {
                message = "Top face not verified: " + rejected + ". Raw image saved: calibration_face_rejected.png";
                cv::imwrite("calibration_face_rejected.png", frame);
                return;
            }
            if (stable < STABLE_FRAMES) {
                message = "COLLECTING: " + std::to_string(stable) + "/20 fresh frames";
                return;
            }
            if (!gridQualityReady) {
                message = windowDrift > DRIFT_LIMIT_PX
                    ? "BLOCKED: window drift " + fmt(windowDrift, 1) + "px exceeds " +
                    fmt(DRIFT_LIMIT_PX, 1) + "px; keep camera/blocks still"
                    : "BLOCKED: noise spread " + fmt(stableSpread, 1) + "px exceeds limit " +
                    fmt(noiseLimitPx, 1) + "px; adjust Noise +/- or improve detection";
                return;
            }
            if (!labelsOK || !refsOK || !planeOK) {
                message = "BLOCKED: confirm label direction, robot references, and same cube-top plane (V / W / H).";
                return;
            }
            auto fitPoints = grid;
            auto mean = averagedCentres();
            if (pointHistory.size() < STABLE_FRAMES || mean.size() != 9) {
                message = "Need 20 fresh stable frames before fitting";
                return;
            }
            for (int i = 0; i < 9; ++i) fitPoints[i].centre = mean[i];
            Mapping candidate;
            if (!makeMapping(fitPoints, refs, frame.size(), candidate)) {
                valid = false;
                message = "REJECTED: mapping could not be calculated; check point ordering and reference geometry";
                return;
            }
            fitReview = true;
            lastFitMax = candidate.max;
            for (int i = 0; i < 9; ++i) {
                fitPixels[i] = fitPoints[i].centre;
                fitDelta[i] = project(candidate.h, fitPoints[i].centre) - refs[i];
                fitResidual[i] = cv::norm(fitDelta[i]);
            }
            if (candidate.max > opt.fitLimit) {
                valid = false;
                int worst = 0;
                double worstError = -1;
                for (int i = 0; i < 9; ++i) {
                    double e = cv::norm(project(candidate.h, fitPoints[i].centre) - refs[i]);
                    if (e > worstError) {
                        worstError = e;
                        worst = i;
                    }
                }
                message = "REJECTED: max " + fmt(candidate.max, 2) + "mm at P" + std::to_string(worst + 1) + ", RMS " +
                    fmt(candidate.rms, 2) + "mm (limit " + fmt(opt.fitLimit, 2) +
                    "). Reference mismatch; see dX/dY table. Not a workspace-boundary error.";
                try {
                    auto dir = sessionDir();
                    std::string tag = "rejected_fit_" + std::to_string(++snapshotNo);
                    std::ofstream csv(dir + "/" + tag + ".csv");
                    if (!csv) throw std::runtime_error("cannot write rejection CSV");
                    csv << "label,pixel_u,pixel_v,reference_x_mm,reference_y_mm,fitted_x_mm,fitted_y_mm,delta_x_mm,delta_y_mm,residual_mm\n";
                    for (int i = 0; i < 9; ++i) {
                        auto predicted = project(candidate.h, fitPoints[i].centre);
                        csv << "P" << i + 1 << ',' << fitPoints[i].centre.x << ',' << fitPoints[i].centre.y << ','
                            << refs[i].x << ',' << refs[i].y << ',' << predicted.x << ',' << predicted.y << ','
                            << predicted.x - refs[i].x << ',' << predicted.y - refs[i].y << ','
                            << cv::norm(predicted - refs[i]) << '\n';
                    }
                    csv.flush();
                    if (!csv) throw std::runtime_error("rejection CSV write failed");
                    if (!cv::imwrite(dir + "/" + tag + "_raw.png", frame))
                        throw std::runtime_error("cannot write rejection image");
                    std::cout << message << " Evidence: " << dir << "/" << tag << ".csv" << std::endl;
                }
                catch (const std::exception& e) {
                    message += " (evidence save failed)";
                    std::cerr << e.what() << std::endl;
                }
                return;
            }
            mapping = candidate;
            valid = true;
            v11TopFaceCalibration = true;
            session.clear();
            try {
                saveCalibration();
                savePersistentCalibration();
                message =
                    "Camera map saved for BOTH Static and Dynamic; reused across cycles/restarts. Test independent positions before motion.";
            }
            catch (const std::exception& e) {
                valid = false;
                message = std::string("SAVE FAILED: ") + e.what();
            }
        }
        void saveCalibration() {
            auto dir = sessionDir();
            cv::FileStorage f((dir + "/" + "calibration.yml"), cv::FileStorage::WRITE);
            if (!f.isOpened()) throw std::runtime_error("Cannot write calibration.yml");
            f << "schema" << 1 << "source" << sourceName() << "simulation" << int(opt.source == "demo")
                << "physically_validated" << 0 << "lens_undistorted" << 0 << "time_local" << stamp() << "width"
                << frame.cols << "height" << frame.rows;
            f << "H_pixel_to_robot_xy" << cv::Mat(mapping.h) << "fit_rms_mm" << mapping.rms << "fit_max_mm"
                << mapping.max << "fit_gate_mm" << opt.fitLimit;
            f << "operator_confirmed_labels" << int(labelsOK) << "operator_confirmed_robot_references" << int(refsOK)
                << "operator_confirmed_same_top_plane" << int(planeOK) << "flip_rows" << int(flipRows) << "flip_columns"
                << int(flipCols);
            f << "hsv_min_saturation" << cfg.sat << "hsv_min_value" << cfg.val << "min_contour_area_px" << cfg.minArea;
            f << "hue_centres" << "[";
            for (auto h : cfg.hue) f << h;
            f << "]" << "hue_tolerances" << "[";
            for (auto t : cfg.tol) f << t;
            f << "]";
            f << "averaged_frames" << int(pointHistory.size()) << "raw_jump_reset_px" << RESET_JUMP_PX
                << "window_90pct_spread_limit_px" << noiseLimitPx << "window_drift_limit_px" << DRIFT_LIMIT_PX
                << "centre_estimator" << "coordinate_median";
            f << "cube_height_nominal_mm" << 35.0 << "points" << "[";
            for (int i = 0; i < 9; ++i)
                f << "{" << "label" << "P" + std::to_string(i + 1) << "pixel_u" << mapping.pixels[i].x << "pixel_v"
                << mapping.pixels[i].y << "robot_x_mm" << refs[i].x << "robot_y_mm" << refs[i].y << "}";
            f << "]";
            f.release();
            std::ofstream csv(dir + "/" + "calibration_points.csv");
            if (!csv) throw std::runtime_error("Cannot write points CSV");
            csv << "label,pixel_u,pixel_v,reference_robot_x_mm,reference_robot_y_mm,predicted_x_mm,predicted_y_mm,fit_error_mm\n"
                << std::setprecision(12);
            for (int i = 0; i < 9; ++i) {
                auto p = project(mapping.h, mapping.pixels[i]);
                csv << "P" << i + 1 << ',' << mapping.pixels[i].x << ',' << mapping.pixels[i].y << ',' << refs[i].x
                    << ',' << refs[i].y << ',' << p.x << ',' << p.y << ',' << cv::norm(p - refs[i]) << '\n';
            }
            cv::imwrite((dir + "/" + "calibration_raw.png"), frame);
            draw();
            cv::imwrite((dir + "/" + "calibration_ui.png"), ui);
        }
        void savePersistentCalibration() {
            cv::FileStorage f("calibration_saved.yml", cv::FileStorage::WRITE);
            if (!f.isOpened()) throw std::runtime_error("Cannot write calibration_saved.yml");
            f << "top_face_estimator"
                << (v11TopFaceCalibration ? (v13Optics().enabled ? "v13_centre_line_v3" : "v11_painted_face_v2")
                    : "legacy");
            f << "schema" << 2 << "width" << mapping.resolution.width << "height" << mapping.resolution.height << "H"
                << cv::Mat(mapping.h) << "rms" << mapping.rms << "max" << mapping.max;
            f << "flip_rows" << int(flipRows) << "flip_columns" << int(flipCols) << "sat" << cfg.sat << "val" << cfg.val
                << "min_area" << cfg.minArea;
            f << "hue" << "[";
            for (auto h : cfg.hue) f << h;
            f << "]" << "tol" << "[";
            for (auto t : cfg.tol) f << t;
            f << "]";
            f << "pixels" << "[";
            for (auto& q : mapping.pixels) f << q.x << q.y;
            f << "]" << "robot" << "[";
            for (auto& q : mapping.robot) f << q.x << q.y;
            f << "]";
            f.release();
            cv::FileStorage state("calibration_validity.yml", cv::FileStorage::WRITE);
            if (!state.isOpened()) throw std::runtime_error("Cannot persist calibration validity");
            state << "valid" << 1;
        }
        // Reload the last accepted fit so the operator does not recalibrate on every start.
        bool loadPersistentCalibration() {
            if (!fileutil::exists("calibration_saved.yml")) return false;
            try {
                if (fileutil::exists("calibration_validity.yml")) {
                    cv::FileStorage state("calibration_validity.yml", cv::FileStorage::READ);
                    int accepted = 0;
                    state["valid"] >> accepted;
                    if (accepted != 1) {
                        message = "Calibration was invalidated. Fit + Save and recapture clear reference.";
                        return false;
                    }
                }
                cv::FileStorage f("calibration_saved.yml", cv::FileStorage::READ);
                if (!f.isOpened()) return false;
                string estimator;
                f["top_face_estimator"] >> estimator;
                v11TopFaceCalibration =
                    estimator == (v13Optics().enabled ? "v13_centre_line_v3" : "v11_painted_face_v2");
                int w = 0, h = 0;
                cv::Mat hm;
                std::vector<double> px, rb;
                std::vector<int> hue, tol;
                Mapping m;
                int sat = 0, val = 0, minArea = 0, fr = 0, fc = 0;
                f["width"] >> w;
                f["height"] >> h;
                f["H"] >> hm;
                f["pixels"] >> px;
                f["robot"] >> rb;
                f["hue"] >> hue;
                f["tol"] >> tol;
                f["rms"] >> m.rms;
                f["max"] >> m.max;
                f["sat"] >> sat;
                f["val"] >> val;
                f["min_area"] >> minArea;
                f["flip_rows"] >> fr;
                f["flip_columns"] >> fc;
                if (w <= 0 || h <= 0 || hm.rows != 3 || hm.cols != 3 || px.size() != 18 || rb.size() != 18 ||
                    hue.size() != 3 || tol.size() != 3)
                    return false;
                hm.convertTo(hm, CV_64F);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) {
                        m.h(r, c) = hm.at<double>(r, c);
                        if (!std::isfinite(m.h(r, c))) return false;
                    }
                for (int i = 0; i < 9; ++i) {
                    m.pixels.emplace_back(px[2 * i], px[2 * i + 1]);
                    m.robot.emplace_back(rb[2 * i], rb[2 * i + 1]);
                }
                m.resolution = cv::Size(w, h);
                for (int i = 0; i < 9; ++i)
                    if (cv::norm(project(m.h, m.pixels[i]) - m.robot[i]) > 5.0) return false;
                mapping = m;
                refs = m.robot;
                flipRows = fr != 0;
                flipCols = fc != 0;
                cfg.sat = sat;
                cfg.val = val;
                cfg.minArea = minArea;
                for (int i = 0; i < 3; ++i) {
                    cfg.hue[i] = hue[i];
                    cfg.tol[i] = tol[i];
                }
                valid = true;
                labelsOK = refsOK = planeOK = true;
                message = "Loaded saved calibration (" + std::to_string(w) + "x" + std::to_string(h) + ", fit max " +
                    fmt(m.max, 2) + " mm). No recalibration needed unless the camera moved.";
                if (!v11TopFaceCalibration)
                    message =
                    "Legacy centre calibration loaded: redo Fit + Save with35mm cubes before Dynamic/Test motion.";
                return true;
            }
            catch (...) {
                return false;
            }
        }
        void captureDetectionFrame() {
            DetectionDiagnostics diag;
            detect(frame, activeColourModel(), &diag);
            int index = 31 - diagnosticRemaining;
            std::string prefix = diagnosticDir + "/frame_" + std::to_string(index);
            if (!cv::imwrite(prefix + "_raw.png", frame)) throw std::runtime_error("Diagnostic raw save failed");
            for (int k = 0; k < 3; ++k) {
                if (!cv::imwrite(prefix + "_mask_" + std::to_string(k) + ".png", diag.thresholdMasks[k]) ||
                    !cv::imwrite(prefix + "_clean_" + std::to_string(k) + ".png", diag.cleanMasks[k]))
                    throw std::runtime_error("Diagnostic mask save failed");
            }
            for (int k = 0; k < 3; ++k)
                if (!diag.fallbackThresholdMasks[k].empty()) {
                    if (!cv::imwrite(prefix + "_fallback_mask_" + std::to_string(k) + ".png",
                        diag.fallbackThresholdMasks[k]) ||
                        !cv::imwrite(prefix + "_fallback_clean_" + std::to_string(k) + ".png",
                            diag.fallbackCleanMasks[k]))
                        throw std::runtime_error("Fallback mask save failed");
                }
            std::ofstream f(prefix + "_contours.csv");
            if (!f) throw std::runtime_error("Diagnostic contour CSV failed");
            f << "colour_0R_1G_2B,x_px,y_px,width_px,height_px,area_px,solidity,result\n";
            for (auto r : diag.rows) f << r << '\n';
            --diagnosticRemaining;
            message = diagnosticRemaining ? "DETECTION CAPTURE: " + std::to_string(diagnosticRemaining) +
                " frames remaining / no PC motion"
                : "DETECTION CAPTURE SAVED: " + diagnosticDir;
            if (!diagnosticRemaining) std::cout << message << std::endl;
        }
        void beginDetectionCapture() {
            if (freeze || frame.empty()) {
                message = "Resume live camera before detection capture";
                return;
            }
            auto dir = sessionDir();
            diagnosticDir = dir + "/detection_" + std::to_string(++snapshotNo);
            fileutil::createDirectory(diagnosticDir);
            std::ofstream f(diagnosticDir + "/settings.csv");
            if (!f) throw std::runtime_error("Cannot save detector settings");
            const auto& model = activeColourModel();
            f << "colour,hue,tolerance,min_sat,min_value,min_area,static_style\n";
            for (int k = 0; k < 3; ++k)
                f << k << ',' << model.hue[k] << ',' << model.tol[k] << ',' << model.sat << ',' << model.val << ','
                << model.minArea << ',' << int(model.staticStyle) << '\n';
            std::ofstream light(diagnosticDir + "/lighting.txt");
            light << lighting.text() << "\nRAW HSV; lighting monitor only. No brightness compensation.\n";
            diagnosticRemaining = 30;
            message = "DETECTION CAPTURE STARTED: keep flashing blocks still. PC motion blocked for 30 fresh frames.";
        }
        void snapshot() {
            auto dir = sessionDir();
            std::string tag = "capture_" + std::to_string(++snapshotNo);
            if (!cv::imwrite((dir + "/" + (tag + "_raw.png")), frame) ||
                !cv::imwrite((dir + "/" + (tag + "_ui.png")), ui))
                throw std::runtime_error("Snapshot failed");
            message = "Saved raw + annotated snapshot in " + dir;
        }
        void compareNine() {
            if (!valid || !test) {
                message = "Save calibration, open Coordinate Test, then use 9-point comparison";
                return;
            }
            auto ordered = detections;
            if (!gridOrder(ordered, flipRows, flipCols)) {
                message =
                    "9-point check requires the original nine-block arrangement; restore all nine blocks to P1-P9";
                return;
            }
            comparison.clear();
            for (int i = 0; i < 9; ++i) {
                auto predicted = project(mapping.h, ordered[i].centre);
                comparison.push_back({ refs[i], predicted, cv::norm(predicted - refs[i]) });
            }
            try {
                auto dir = sessionDir();
                std::string tag = "nine_point_comparison_" + std::to_string(++snapshotNo);
                comparisonPath = dir + "/" + tag + ".csv";
                std::ofstream csv(comparisonPath);
                if (!csv) throw std::runtime_error("Cannot write comparison CSV");
                csv << "label,code_x_mm,code_y_mm,camera_x_mm,camera_y_mm,delta_x_mm,delta_y_mm,xy_difference_mm,pixel_u,pixel_v\n"
                    << std::setprecision(12);
                for (int i = 0; i < 9; ++i) {
                    auto r = comparison[i];
                    auto delta = r.predicted - r.reference;
                    csv << "P" << i + 1 << ',' << r.reference.x << ',' << r.reference.y << ',' << r.predicted.x << ','
                        << r.predicted.y << ',' << delta.x << ',' << delta.y << ',' << r.error << ','
                        << ordered[i].centre.x << ',' << ordered[i].centre.y << '\n';
                }
                csv.flush();
                if (!csv) throw std::runtime_error("Comparison write failed");
                if (!cv::imwrite(dir + "/" + tag + "_raw.png", frame))
                    throw std::runtime_error("Comparison snapshot failed");
                comparisonView = true;
                freeze = true;
                message =
                    "9-point comparison saved/frozen. X returns live. These are calibration-reference differences, not independent accuracy.";
            }
            catch (const std::exception& e) {
                message = e.what();
            }
        }
        void beginMeasurement() {
            if (!test || !valid || detections.empty() || !selectStable) {
                message = "Need a valid mapping and a stationary selected block on Coordinate Test.";
                return;
            }
            auto p = selectedStability.ready ? selectedStability.centre : detections[selected].centre;
            auto pred = project(mapping.h, p);
            if (!inRegion(mapping, pred) && !enlargedValidation && !enlargedCoverage(pred)) {
                message = "Outside calibrated region: enable MEASURE AREA (no motion) to record actual XY.";
                return;
            }
            measuredPixel = p;
            measuredPrediction = pred;
            validationFrame = frame.clone();
            input.clear();
            entering = true;
            freeze = true;
            message = "Enter independently measured ABB X,Y in mm; Enter saves, Esc cancels.";
        }
        void finishMeasurement() {
            std::istringstream s(input);
            double x, y;
            char comma;
            std::string extra;
            if (!(s >> x >> comma >> y) || comma != ',' || (s >> extra) || !std::isfinite(x) || !std::isfinite(y) ||
                std::abs(x) > 10000 || std::abs(y) > 10000) {
                message = "Invalid value. Example: 420.50,-12.25 . Esc cancels.";
                return;
            }
            Point2d actual{ x, y };
            double error = cv::norm(actual - measuredPrediction);
            auto dir = sessionDir();
            auto path = dir + "/" + "independent_checks.csv";
            bool exists = fileutil::exists(path);
            std::ofstream f(path, std::ios::app);
            if (!f) throw std::runtime_error("Cannot write validation CSV");
            if (!exists)
                f << "time_local,source,pixel_u,pixel_v,predicted_x_mm,predicted_y_mm,measured_x_mm,measured_y_mm,error_mm,evidence_image\n";
            std::string filename = "check_" + std::to_string(++snapshotNo) + "_raw.png";
            cv::imwrite((dir + "/" + filename), validationFrame);
            f << std::setprecision(12) << stamp() << ',' << opt.source << ',' << measuredPixel.x << ','
                << measuredPixel.y << ',' << measuredPrediction.x << ',' << measuredPrediction.y << ',' << x << ',' << y
                << ',' << error << ',' << filename << '\n';
            if (enlargedValidation) {
                if (!enlargedMatches()) {
                    enlargedPredicted.clear();
                    enlargedActual.clear();
                    enlargedHull.clear();
                    enlargedMapping = mapping.h;
                    enlargedResolution = frame.size();
                }
                enlargedApproved = false;
                int replace = -1;
                for (size_t i = 0; i < enlargedPredicted.size(); ++i)
                    if (cv::norm(enlargedPredicted[i] - measuredPrediction) < 25) replace = int(i);
                if (replace >= 0) {
                    enlargedPredicted[replace] = measuredPrediction;
                    enlargedActual[replace] = actual;
                }
                else {
                    enlargedPredicted.push_back(measuredPrediction);
                    enlargedActual.push_back(actual);
                }
                saveEnlarged();
            }
            entering = false;
            freeze = false;
            message = "Independent check saved: error " + fmt(error) +
                " mm. This does not automatically approve robot motion.";
        }
        void sampleAt(cv::Point p) {
            if (sample < 0 || frame.empty()) return;
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            cv::Rect patch(p.x - 5, p.y - 5, 11, 11);
            patch &= cv::Rect(0, 0, frame.cols, frame.rows);
            double x = 0, y = 0;
            int count = 0;
            for (int r = patch.y; r < patch.br().y; ++r)
                for (int c = patch.x; c < patch.br().x; ++c) {
                    auto a = hsv.at<cv::Vec3b>(r, c);
                    if (a[1] > 50 && a[2] > 40) {
                        double angle = a[0] * CV_PI / 90.;
                        x += std::cos(angle);
                        y += std::sin(angle);
                        ++count;
                    }
                }
            if (count < 10) {
                message = "Sample failed: click a clear coloured top face.";
                return;
            }
            double angle = std::atan2(y, x);
            if (angle < 0) angle += 2 * CV_PI;
            cfg.hue[sample] = int(std::round(angle * 90 / CV_PI)) % 180;
            cfg.tol[sample] = 13;
            if (test && useStaticDetector) {
                staticCfg.hue[sample] = cfg.hue[sample];
                staticCfg.tol[sample] = 13;
                staticColourEdited = true;
            }
            sample = -1;
            invalidate("Colour sampled. Check detections and reconfirm calibration.");
        }
        void action(char k) {
            switch (k) {
            case '1':
                test = false;
                planning = false;
                planLocked = false;
                entering = false;
                freeze = false;
                message = "Calibration page. Arrange nine cubes and verify labels.";
                break;
            case '2':
                if (!valid) {
                    message = "Fit and save calibration before Coordinate Test.";
                    break;
                }
                test = true;
                planning = false;
                planLocked = false;
                freeze = false;
                testStable = 0;
                selectStable = false;
                selectedStability.clear();
                message = "Remove grid without moving camera/support plane. Test new block positions.";
                break;
            case '3':
                if (!valid) {
                    message = "Calibrate before Planning Preview.";
                    break;
                }
                test = true;
                planning = true;
                freeze = false;
                planStable = 0;
                planAnchor.clear();
                planLocked = false;
                message = "Planning only. A/D choose S1/S2/S3, P lock, U unlock, L save CSV.";
                break;
            case 'a':
                if (planning) {
                    pattern = pattern == 1 ? 3 : pattern - 1;
                    planLocked = false;
                }
                break;
            case 'd':
                if (planning) {
                    pattern = pattern == 3 ? 1 : pattern + 1;
                    planLocked = false;
                }
                break;
            case 'p':
                if (planning) {
                    auto p = currentPlan();
                    if (p.valid) {
                        lockedPlan = p;
                        planLocked = true;
                        message = "Preview locked. Scene changes automatically unlock it. No commands sent.";
                    }
                    else
                        message = p.reason;
                }
                break;
            case 'u':
                planLocked = false;
                message = "Preview unlocked.";
                break;
            case 'l':
                savePlan();
                break;
            case '-':
            case '+':
            case '=':
                noiseLimitPx = std::max(1.0, std::min(8.0, noiseLimitPx + (k == '-' ? -.5 : .5)));
                valid = false;
                gridQualityReady = false;
                stable = 0;
                previous.clear();
                pointHistory.clear();
                lastRawCentres.clear();
                message = "Noise limit " + fmt(noiseLimitPx, 1) +
                    "px: collecting new frames; fit/physical validation still required";
                break;
            case 'x':
                if (comparisonView) {
                    comparisonView = false;
                    freeze = false;
                    message = "Returned to live coordinate view";
                }
                else
                    compareNine();
                break;
            case 't':
                rawView = !rawView;
                message = rawView ? "Raw live image: overlays hidden (no stabilisation/warping)"
                    : "Median markers from fresh frames; motion gate checks raw spread and drift";
                break;
            case 'c':
                calibration();
                break;
            case 'i':
                invalidate("Mapping invalidated. Recalibrate after camera or fixture movement.");
                break;
            case 'v':
                labelsOK = !labelsOK;
                if (!labelsOK) valid = false;
                message = "P1-P9 direction confirmation changed.";
                break;
            case 'w':
                refsOK = !refsOK;
                if (!refsOK) valid = false;
                message = "Confirm ONLY after robot reference XY has been physically checked.";
                break;
            case 'h':
                planeOK = !planeOK;
                if (!planeOK) valid = false;
                message = "Confirm same 35 mm block-top plane and unchanged support height.";
                break;
            case 'f':
                if (!test) {
                    flipRows = !flipRows;
                    invalidate("Row order reversed. Recheck P1-P9.");
                }
                break;
            case 'j':
                if (!test) {
                    flipCols = !flipCols;
                    invalidate("Column order reversed. Recheck P1-P9.");
                }
                break;
            case 'r':
                sample = 0;
                message = "Click RED top face to sample hue.";
                break;
            case 'g':
                sample = 1;
                message = "Click GREEN top face to sample hue.";
                break;
            case 'b':
                sample = 2;
                message = "Click BLUE top face to sample hue.";
                break;
            case 'k':
                cfg = ColourModel{};
                if (test && useStaticDetector) {
                    staticCfg = ColourModel{};
                    staticCfg.staticStyle = true;
                    staticCfg.sat = 65;
                    staticCfg.val = 60;
                    staticCfg.tol = { {12, 15, 15} };
                    staticColourEdited = true;
                }
                invalidate("Default colour thresholds restored.");
                break;
            case 's':
                if (test && !planning)
                    beginDetectionCapture();
                else
                    snapshot();
                break;
            case 'e':
                beginMeasurement();
                break;
            case ' ':
                freeze = !freeze;
                message = freeze ? "Frame frozen; do not calibrate a moved camera from stale imagery."
                    : "Live frames resumed.";
                break;
            case 'n':
                if (opt.source == "demo") {
                    block = { 55. + ((++trial * 47) % 115), 55. + ((trial * 31) % 110) };
                }
                break;
            case 'm':
                if (opt.source == "demo") {
                    pose = (pose + 1) % 3;
                    invalidate("Synthetic camera changed. Return to Calibration and fit again.");
                }
                break;
            case 'o':
                if (opt.source == "demo") {
                    missing = !missing;
                    invalidate("Synthetic P5 visibility changed.");
                }
                break;
            case 'q':
                exit = true;
                break;
            }
        }
        void click(int x, int y) {
            if (entering) return;
            if (test && !planning && cv::Rect(865, 620, 380, 32).contains({ x, y })) {
                action('x');
                return;
            }
            if (!test && !planning && cv::Rect(1110, 590, 50, 32).contains({ x, y })) {
                action('-');
                return;
            }
            if (!test && !planning && cv::Rect(1170, 590, 50, 32).contains({ x, y })) {
                action('+');
                return;
            }
            for (auto& b : buttons)
                if (b.r.contains({ x, y })) {
                    action(b.action);
                    return;
                }
            if (x >= 855 && y >= 350 && y < 435) {
                char key = y < 385 ? 'v' : y < 410 ? 'w' : 'h';
                action(key);
                return;
            }
            if (planning && x >= 865 && x < 1255 && y >= 430 && y < 464) {
                pattern = std::min(3, 1 + (x - 865) / 130);
                planLocked = false;
                return;
            }
            if (!VIDEO.contains({ x, y })) return;
            double scale = std::min(double(VIDEO.width) / frame.cols, double(VIDEO.height) / frame.rows);
            int ox = VIDEO.x + int((VIDEO.width - frame.cols * scale) / 2),
                oy = VIDEO.y + int((VIDEO.height - frame.rows * scale) / 2);
            cv::Point p(int((x - ox) / scale), int((y - oy) / scale));
            if (p.x < 0 || p.y < 0 || p.x >= frame.cols || p.y >= frame.rows) return;
            if (sample >= 0) {
                sampleAt(p);
                return;
            }
            if (test && !detections.empty()) {
                double best = 1e12;
                for (int i = 0; i < int(detections.size()); ++i) {
                    double d = cv::norm(detections[i].centre - Point2d(p));
                    if (d < best) {
                        best = d;
                        selected = i;
                    }
                }
                bool changed = !selectionTracked || selectionMissing ||
                    selectionColour != detections[selected].colour ||
                    cv::norm(selectionPosition - detections[selected].centre) > 12;
                if (changed) {
                    selectedStability.clear();
                    testStable = 0;
                    selectStable = false;
                }
                selectionTracked = true;
                selectionMissing = false;
                selectionPosition = detections[selected].centre;
                selectionColour = detections[selected].colour;
            }
        }
        void key(int key) {
            if (key < 0) return;
            key &= 255;
            if (entering) {
                if (key == 27) {
                    entering = false;
                    freeze = false;
                    message = "Measurement cancelled.";
                }
                else if (key == 13 || key == 10)
                    finishMeasurement();
                else if (key == 8) {
                    if (!input.empty()) input.pop_back();
                }
                else if ((key >= '0' && key <= '9') || key == '-' || key == '+' || key == '.' || key == ',' ||
                    key == ' ') {
                    if (input.size() < 50) input += char(key);
                }
                return;
            }
            if (key == 27) {
                exit = true;
                return;
            }
            action(char(std::tolower(key)));
        }
        void draw() {
            ui = cv::Mat(UIH, UIW, CV_8UC3, BG);
            auto text = [&](int x, int y, const std::string& s, double scale = .48, cv::Scalar color = FG) {
                cv::putText(ui, s, { x, y }, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
                };
            text(18, 27, "HAND-EYE / CALIBRATION + PLANNING", .64);
            text(760, 26, sourceName() + " | NO SERIAL / NO ROBOT MOTION", .47, GOLD);
            for (auto& b : buttons) {
                cv::rectangle(ui, b.r,
                    ((planning && b.action == '3') || (test && !planning && b.action == '2') ||
                        (!test && b.action == '1'))
                    ? cv::Scalar(100, 75, 45)
                    : PANEL,
                    cv::FILLED);
                text(b.r.x + 8, b.r.y + 23, b.label, .46);
            }
            cv::Mat annotated = frame.clone();
            auto source = test ? detections : (grid.empty() ? detections : grid);
            auto means = averagedCentres();
            for (int i = 0; i < int(source.size()); ++i) {
                auto d = source[i];
                if (!test && grid.size() == 9 && means.size() == 9) d.centre = means[i];
                cv::Scalar color = d.colour == 0 ? cv::Scalar(65, 85, 230)
                    : d.colour == 1 ? cv::Scalar(70, 210, 90)
                    : cv::Scalar(230, 150, 65);
                cv::polylines(annotated, d.quad, true, color, 2);
                cv::drawMarker(annotated, d.centre, color, cv::MARKER_CROSS, 14, 2);
                std::string tag = test ? "B" + std::to_string(i + 1) : grid.empty() ? "?" : "P" + std::to_string(i + 1);
                if (test && valid) {
                    auto p = project(mapping.h, d.centre);
                    tag += " X" + fmt(p.x, 1) + " Y" + fmt(p.y, 1);
                    if (!reachOk(p)) tag += " XY LIMIT";
                }
                cv::putText(annotated, tag, { d.box.x, std::max(16, d.box.y - 7) }, cv::FONT_HERSHEY_SIMPLEX, .45, color,
                    1, cv::LINE_AA);
                if (test && i == selected && !selectionMissing)
                    cv::rectangle(annotated, d.box, cv::Scalar(0, 220, 255), 2);
            }
            if (valid) {
                std::vector<cv::Point> poly;
                for (int i : {0, 2, 8, 6}) {
                    auto p = project(mapping.h.inv(), mapping.robot[i]);
                    poly.emplace_back(cvRound(p.x), cvRound(p.y));
                }
                cv::polylines(annotated, poly, true, cv::Scalar(0, 220, 255), 2);
            }
            else if (!test && grid.size() == 9) {
                std::vector<cv::Point> poly;
                for (int i : {0, 2, 8, 6}) poly.emplace_back(cvRound(grid[i].centre.x), cvRound(grid[i].centre.y));
                cv::polylines(annotated, poly, true, cv::Scalar(120, 170, 180), 1);
            }
            if (enlargedApproved && enlargedMatches()) {
                std::vector<cv::Point> poly;
                for (auto q : enlargedHull) {
                    auto p = project(mapping.h.inv(), Point2d(q.x, q.y));
                    poly.emplace_back(cvRound(p.x), cvRound(p.y));
                }
                cv::polylines(annotated, poly, true, cv::Scalar(230, 210, 60), 2);
            }
            if (rawView) annotated = frame.clone();
            double scale = std::min(double(VIDEO.width) / frame.cols, double(VIDEO.height) / frame.rows);
            cv::Mat resized;
            cv::resize(annotated, resized, {}, scale, scale);
            int ox = VIDEO.x + (VIDEO.width - resized.cols) / 2, oy = VIDEO.y + (VIDEO.height - resized.rows) / 2;
            resized.copyTo(ui(cv::Rect(ox, oy, resized.cols, resized.rows)));
            cv::rectangle(ui, { 850, 100, 415, 615 }, PANEL, cv::FILLED);
            int y = 125;
            auto row = [&](std::string s, cv::Scalar col = FG) {
                text(865, y, s, .46, col);
                y += 25;
                };
            row(planning ? "PLANNING PREVIEW" : test ? "COORDINATE TEST" : "CALIBRATION", GOLD);
            row("Tops: " + std::to_string(detections.size()) + "  R:" +
                std::to_string(std::count_if(detections.begin(), detections.end(),
                    [](const Detection& d) { return d.colour == 0; })) +
                " G:" +
                std::to_string(std::count_if(detections.begin(), detections.end(),
                    [](const Detection& d) { return d.colour == 1; })) +
                " B:" + std::to_string(std::count_if(detections.begin(), detections.end(), [](const Detection& d) {
                    return d.colour == 2;
                    })));
            row(planning ? "Stable scene: " + std::to_string(std::min(planStable, STABLE_FRAMES)) + " / 20 frames"
                : test
                ? "Selected stable: " + std::to_string(std::min(testStable, 8)) + " / 8 clustered samples (last 12)"
                : "Stable grid: " + std::to_string(std::min(stable, STABLE_FRAMES)) + " / 20 frames");
            row("Source resolution: " + std::to_string(frame.cols) + " x " + std::to_string(frame.rows));
            row(valid ? "Shared map: SAVED (Static + Dynamic)"
                : fitReview && lastFitMax > opt.fitLimit ? "Last fit: REJECTED - see residuals below"
                : gridQualityReady ? "Mapping: READY TO FIT"
                : "Mapping: NOT READY");
            if (valid) {
                row("Fit RMS " + fmt(mapping.rms) + " / max " + fmt(mapping.max) + " mm");
            }
            else
                row("Fit gate: max " + fmt(opt.fitLimit) + " mm (setup only)");
            row("Lens distortion correction: NONE", GOLD);
            row(enlargedHoverView ? "Area: ENLARGED HOVER (test only)"
                : enlargedValidation ? "MEASURE ONLY / extrapolation visible"
                : enlargedApproved && enlargedMatches() ? "Area: original + measured XY envelope"
                : "Area: calibrated region only");
            text(865, 330, "Raw step " + fmt(rawPointStep, 1) + " px | drift " + fmt(rawPointDrift, 1) + " px", .42,
                GOLD);
            text(865, 345, "Spread " + fmt(stableSpread, 1) + " / drift " + fmt(windowDrift, 1) + " px", .36, MUTED);
            y = 370;
            row(std::string(labelsOK ? "[x] " : "[ ] ") + "V  P1-P9 labels/direction checked");
            row(std::string(refsOK ? "[x] " : "[ ] ") + "W  Robot reference XY verified");
            row(std::string(planeOK ? "[x] " : "[ ] ") + "H  Same 35 mm block-top plane");
            y = 445;
            if (planning) {
                for (int s = 1; s <= 3; ++s) {
                    cv::Rect r(865 + (s - 1) * 130, 430, 120, 34);
                    cv::rectangle(ui, r, s == pattern ? cv::Scalar(100, 75, 45) : BG, cv::FILLED);
                    text(r.x + 30, 453, "S" + std::to_string(s), .52);
                }
                y = 487;
                Plan p = currentPlan();
                row("Target S" + std::to_string(pattern) + " (fixed stack)", GOLD);
                for (int l = 0; l < 3; ++l) {
                    int color = PATTERNS[pattern - 1][2 - l];
                    row("L" + std::to_string(l + 1) + " " + COLOUR_NAMES[color] +
                        (l == 0 ? " (bottom)"
                            : l == 2 ? " (top)"
                            : ""));
                    if (p.valid)
                        row(" B" + std::to_string(p.source[l] + 1) + " X" + fmt(p.xy[l].x, 1) + " Y" +
                            fmt(p.xy[l].y, 1),
                            MUTED);
                }
                row(p.valid ? (planLocked ? "LOCKED / PREVIEW ONLY" : "LIVE / PREVIEW ONLY") : p.reason, GOLD);
                text(865, 701, "A/D pattern | P lock | U unlock | L save", .40, MUTED);
            }
            else if (!test) {
                row("F reverse rows / J reverse columns");
                row("Distance checks (mapped vs reference):");
                if (valid && grid.size() == 9) {
                    for (auto pair : std::vector<std::pair<int, int>>{ {0, 1}, {0, 3}, {6, 8} }) {
                        double px = cv::norm(grid[pair.first].centre - grid[pair.second].centre),
                            nom = cv::norm(refs[pair.first] - refs[pair.second]);
                        double mapped = cv::norm(project(mapping.h, grid[pair.first].centre) -
                            project(mapping.h, grid[pair.second].centre));
                        row("P" + std::to_string(pair.first + 1) + "-P" + std::to_string(pair.second + 1) + ": " +
                            fmt(px, 0) + " px / " + fmt(mapped, 1) + " mm");
                        row("  reference " + fmt(nom, 1) + " mm", MUTED);
                    }
                }
                else
                    row("Fit after confirming all three checks.");
            }
            else if (selectionMissing) {
                row("SELECTED BLOCK LOST", GOLD);
                row("No fresh selected XY; motion blocked");
                row("Reselect only if you moved the block");
            }
            else if (valid && !detections.empty()) {
                auto p = project(mapping.h, detections[selected].centre);
                row("Selected B" + std::to_string(selected + 1) + ": X " + fmt(p.x) + " Y " + fmt(p.y));
                row(enlargedValidation ? "MEASURE ONLY - NO PC MOTION"
                    : (inRegion(mapping, p) || enlargedCoverage(p)) ? "WITHIN XY COVERAGE"
                    : "OUTSIDE VALIDATED XY AREA",
                    GOLD);
                row("Distance from base: " + fmt(std::hypot(p.x, p.y), 0) + " mm | limit 360-540");
                row(selectionMissing ? "SELECTED BLOCK LOST - reselect if moved"
                    : selectStable ? "Selected block stable (fresh frames)"
                    : "Settling: " + std::to_string(std::min(testStable, 8)) + "/8 clustered samples");
                row("B numbers are detections, not P slots.");
                row("E: enter measured robot X,Y.");
                row("Fit error is not independent accuracy.");
            }
            if (test && !planning) {
                cv::Rect r(865, 620, 380, 32);
                cv::rectangle(ui, r, BG, cv::FILLED);
                text(875, 642, comparisonView ? "X Back to live camera" : "X 9-point comparison (grid only)", .43, FG);
            }
            if (comparisonView && comparison.size() == 9) {
                cv::rectangle(ui, VIDEO, BG, cv::FILLED);
                text(30, 135, "P1-P9 COORDINATE COMPARISON / mm", .60, FG);
                text(30, 166, "Original nine-block grid required. Snapshot is frozen.", .43, MUTED);
                text(30, 207, "Point    Code X     Code Y    Camera X   Camera Y     dX       dY      XY diff", .42,
                    GOLD);
                double square = 0, maximum = 0;
                for (int i = 0; i < 9; ++i) {
                    auto r = comparison[i];
                    auto d = r.predicted - r.reference;
                    int y = 243 + i * 39;
                    text(30, y, "P" + std::to_string(i + 1), .48, FG);
                    const double vals[] = { r.reference.x, r.reference.y, r.predicted.x, r.predicted.y,
                                           d.x,           d.y,           r.error };
                    const int cols[] = { 110, 210, 310, 410, 520, 615, 715 };
                    for (int j = 0; j < 7; ++j) text(cols[j], y, fmt(vals[j], 2), .43, j >= 4 ? GOLD : FG);
                    square += r.error * r.error;
                    maximum = std::max(maximum, r.error);
                }
                text(30, 625, "RMS " + fmt(std::sqrt(square / 9), 2) + " mm | max " + fmt(maximum, 2) + " mm", .50, FG);
                text(30, 660, "Difference = sqrt(dX*dX + dY*dY). X returns live.", .43, MUTED);
                text(30, 690, "Not an independent physical accuracy test.", .43, GOLD);
            }
            if (!planning && !test) {
                if (fitReview) {
                    cv::rectangle(ui, cv::Rect(855, 438, 405, 150), PANEL, cv::FILLED);
                    text(865, 451, "LAST FIT: dX / dY / residual (mm)", .36, GOLD);
                    for (int i = 0; i < 9; ++i)
                        text(865, 466 + i * 12,
                            "P" + std::to_string(i + 1) + "   " + fmt(fitDelta[i].x) + " / " + fmt(fitDelta[i].y) +
                            " / " + fmt(fitResidual[i]),
                            .33, fitResidual[i] > opt.fitLimit ? GOLD : FG);
                    text(865, 584, "Reference disagreement; not an OUTSIDE flag", .32, MUTED);
                }
                text(865, 612, "Noise limit: " + fmt(noiseLimitPx, 1) + " px", .42, FG);
                for (int i = 0; i < 2; ++i) {
                    cv::Rect r(1110 + i * 60, 590, 50, 32);
                    cv::rectangle(ui, r, BG, cv::FILLED);
                    text(r.x + 19, r.y + 23, i ? "+" : "-", .55, FG);
                }
                text(865, 640, gridQualityReady ? "READY: verify V/W/H, then Fit + save" : stabilityReason, .37,
                    gridQualityReady ? cv::Scalar(70, 210, 90) : GOLD);
            }
            if (!planning)
                text(865, 665, rawView ? "T: show filtered markers" : "T: raw view / hide overlays", .40, MUTED);
            if (!planning) {
                y = 690;
                text(865, y, freeze ? "FRAME FROZEN" : "SPACE: freeze / resume", .45, freeze ? GOLD : MUTED);
            }
            text(18, 727, lighting.text() + (useStaticDetector ? " / STATIC SHARED DETECTOR" : " / compensation OFF"),
                .40, GOLD);
            text(18, 744,
                "R/G/B then click top face: sample colour | K: default colours | I: invalidate after camera movement",
                .45);
            text(18, 771, "P1-P9 must be checked. Nine-point symmetry cannot identify direction automatically.", .46,
                GOLD);
            text(
                18, 797,
                "Frame limits block CENTRES only. Jaw clearance, Z, rotations, reachability and physical accuracy are not validated.",
                .43, MUTED);
            std::string msg = entering ? "ACTUAL ABB X,Y: " + input + "_" : message;
            int yy = 829;
            while (!msg.empty() && yy < 895) {
                size_t cut = std::min<size_t>(125, msg.size());
                if (cut < msg.size()) {
                    auto pos = msg.rfind(' ', cut);
                    if (pos != std::string::npos && pos > 30) cut = pos;
                }
                text(18, yy, msg.substr(0, cut), .46, GOLD);
                msg.erase(0, cut);
                yy += 23;
            }
        }
    };

    // ------------------------------------------------------------
    //  1e  Synthetic images and calibration self-test
    // ------------------------------------------------------------
    H syntheticCamera(int pose) {
        if (pose == 1) return H(2.5, .25, 110, -.04, -2.2, 610, .00075, -.0002, 1);
        if (pose == 2) return H(2.8, -.1, 90, .1, -2.4, 590, .0002, .0005, 1);
        return H(2.8, .12, 70, .06, -2.5, 620, .00055, .00025, 1);
    }
    cv::Mat synthetic(const App& a) {
        cv::Mat image(720, 960, CV_8UC3, cv::Scalar(205, 205, 205));
        auto draw = [&](Point2d p, int color) {
            std::vector<cv::Point> q;
            for (auto delta : std::vector<Point2d>{ {-18, -18}, {18, -18}, {18, 18}, {-18, 18} }) {
                auto v = project(syntheticCamera(a.pose), p + delta);
                q.emplace_back(cvRound(v.x), cvRound(v.y));
            }
            cv::Scalar colors[] = { cv::Scalar(45, 45, 220), cv::Scalar(45, 180, 45), cv::Scalar(210, 70, 40) };
            cv::fillConvexPoly(image, q, colors[color % 3], cv::LINE_8);
            };
        if (a.planning) {
            draw(a.block, 0);
            draw({ 165, 150 }, 1);
            draw({ 65, 65 }, 2);
        }
        else if (a.test)
            draw(a.block, 0);
        else {
            auto p = localGrid();
            for (int i = 0; i < 9; ++i)
                if (!a.missing || i != 4) draw(p[i], i % 3);
        }
        return image;
    }
    bool loadReferences(const std::string& filename, std::vector<Point2d>& refs) {
        std::ifstream f(filename);
        if (!f) return false;
        std::string line;
        std::getline(f, line);
        std::vector<Point2d> points(9);
        std::array<bool, 9> seen{};
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream s(line);
            std::string name, extra;
            double x, y;
            if (!(s >> name >> x >> y) || (s >> extra) || name.size() != 2 || name[0] != 'P' || name[1] < '1' ||
                name[1] > '9' || !std::isfinite(x) || !std::isfinite(y))
                return false;
            int i = name[1] - '1';
            if (seen[i]) return false;
            seen[i] = true;
            points[i] = { x, y };
        }
        if (!std::all_of(seen.begin(), seen.end(), [](bool a) { return a; }) || polygonArea(points) < 100) return false;
        refs = points;
        return true;
    }
    bool selfTest() {
        {
            cv::Mat dark(240, 300, CV_8UC3, cv::Scalar(130, 130, 130));
            cv::rectangle(dark, { 35, 45, 40, 40 }, cv::Scalar(45, 12, 12), cv::FILLED);
            cv::rectangle(dark, { 110, 45, 40, 40 }, cv::Scalar(12, 45, 12), cv::FILLED);
            cv::rectangle(dark, { 185, 45, 40, 40 }, cv::Scalar(12, 12, 45), cv::FILLED);
            cv::rectangle(dark, { 34, 59, 4, 4 }, cv::Scalar(130, 130, 130), cv::FILLED);
            auto found = detect(dark, ColourModel{});
            if (found.size() != 3) throw std::runtime_error("Dim/notched block detection failed");
            // Rounded corners: full square synthetic fixtures previously missed this regression.
            cv::Mat rounded(220, 300, CV_8UC3, cv::Scalar(120, 120, 120));
            for (int k = 0; k < 3; ++k) {
                cv::Point c(50 + k * 90, 100);
                cv::Scalar colour = k == 0 ? cv::Scalar(0, 0, 160)
                    : k == 1 ? cv::Scalar(0, 160, 0)
                    : cv::Scalar(160, 0, 0);
                cv::rectangle(rounded, cv::Rect(c.x - 15, c.y - 20, 30, 40), colour, cv::FILLED);
                cv::rectangle(rounded, cv::Rect(c.x - 20, c.y - 15, 40, 30), colour, cv::FILLED);
                for (int dx : {-15, 15})
                    for (int dy : {-15, 15}) cv::circle(rounded, c + cv::Point(dx, dy), 5, colour, cv::FILLED);
            }
            if (detect(rounded, ColourModel{}).size() != 3) throw std::runtime_error("Rounded block corners rejected");
            cv::Mat circle(200, 200, CV_8UC3, cv::Scalar(120, 120, 120));
            cv::circle(circle, { 100, 100 }, 30, { 0, 0, 160 }, cv::FILLED);
            if (!detect(circle, ColourModel{}).empty())
                throw std::runtime_error("Circular object accepted as square top");
            std::vector<Detection> missing = found;
            if (gridOrder(missing, false, false)) throw std::runtime_error("Partial dim grid accepted");
        }
        double worst = 0;
        for (int pose = 0; pose < 3; ++pose) {
            App a;
            a.opt.source = "demo";
            a.pose = pose;
            for (int i = 0; i < STABLE_FRAMES + 1; ++i) {
                a.frame = synthetic(a);
                a.update();
            }
            if (a.grid.size() != 9 || a.stable < STABLE_FRAMES)
                throw std::runtime_error("Nine-point automatic detection/stability failed");
            Mapping m;
            if (!makeMapping(a.grid, a.refs, a.frame.size(), m) || m.max > 2) throw std::runtime_error("Fit failed");
            a.mapping = m;
            a.valid = true;
            a.test = true;
            for (auto p : std::vector<Point2d>{ {55, 55}, {100, 93}, {178, 160}, {125, 145}, {180, 50} }) {
                a.block = p;
                a.frame = synthetic(a);
                a.update();
                if (a.detections.size() != 1) throw std::runtime_error("Independent block detection failed");
                auto pred = project(m.h, a.detections[0].centre);
                Point2d actual{ 552.65 - p.y, p.x - 122.8 };
                double e = cv::norm(pred - actual);
                worst = std::max(worst, e);
                if (e > 2 || !inRegion(m, pred)) throw std::runtime_error("Unseen-position mapping failed");
            }
            a.invalidate("test");
            if (a.valid || a.labelsOK || a.refsOK || a.planeOK) throw std::runtime_error("Invalidation failed");
        }
        App a;
        a.opt.source = "demo";
        a.missing = true;
        a.frame = synthetic(a);
        a.update();
        if (!a.grid.empty()) throw std::runtime_error("Missing point accepted");
        H h;
        if (fitH({ {0, 0}, {1, 1}, {2, 2}, {3, 3} }, { {0, 0}, {1, 1}, {2, 2}, {3, 3} }, h))
            throw std::runtime_error("Degenerate fit accepted");
        a.missing = false;
        a.frame = synthetic(a);
        a.update();
        Mapping m;
        if (!makeMapping(a.grid, a.refs, a.frame.size(), m)) throw std::runtime_error("Region test mapping failed");
        if (inRegion(m, { 600, 200 })) throw std::runtime_error("Outside region accepted");
        // Verify smoothing never hides motion or fills missing detections.
        {
            App jitter;
            jitter.opt.source = "demo";
            cv::Mat base = synthetic(jitter);
            for (int i = 0; i < 30; ++i) {
                cv::Matx23d shift(1, 0, i % 2, 0, 1, 0);
                cv::warpAffine(base, jitter.frame, shift, base.size());
                jitter.update();
            }
            if (jitter.stable < 20 || jitter.pointHistory.size() != 20)
                throw std::runtime_error("Small raw jitter did not stabilise");
            cv::Matx23d shift(1, 0, 16, 0, 1, 0);
            cv::warpAffine(base, jitter.frame, shift, base.size());
            jitter.update();
            if (jitter.stable != 1 || jitter.pointHistory.size() != 1)
                throw std::runtime_error("Smoothing concealed actual movement");
            jitter.missing = true;
            jitter.frame = synthetic(jitter);
            jitter.update();
            if (jitter.gridQualityReady || !jitter.grid.empty() || jitter.stable != 1)
                throw std::runtime_error("Brief missing grid did not pause safely");
            for (int n = 0; n < 6; ++n) {
                jitter.frame = synthetic(jitter);
                jitter.update();
            }
            if (jitter.stable || !jitter.pointHistory.empty())
                throw std::runtime_error("Long missing grid retained samples");
        }
        {
            App measured;
            std::vector<Detection> fixture;
            for (int i = 0; i < 9; ++i) {
                Detection d;
                d.centre = { double(60 + i % 3 * 80), double(60 + i / 3 * 80) };
                d.area = 900;
                d.colour = i % 3;
                fixture.push_back(d);
            }
            for (int t = 0; t < 30; ++t) {
                measured.grid = fixture;
                for (auto& d : measured.grid) d.centre.x += (t % 2 ? 1.8 : -1.8);
                measured.updateGridStability();
            }
            if (measured.stable != 20 || !measured.gridQualityReady)
                throw std::runtime_error("Bounded segmentation jitter rejected");
            measured.invalidate("test");
            for (int t = 0; t < 20; ++t) {
                measured.grid = fixture;
                for (auto& d : measured.grid) d.centre.x += t * .4;
                measured.updateGridStability();
            }
            if (measured.gridQualityReady) throw std::runtime_error("Slow physical drift accepted");
            measured.grid = fixture;
            measured.grid.pop_back();
            measured.updateGridStability();
            if (measured.gridQualityReady || !measured.grid.empty() || measured.stable != 20)
                throw std::runtime_error("Missing grid did not pause / disabled Fit");
            measured.grid = fixture;
            measured.updateGridStability();
            if (measured.stable != 20) throw std::runtime_error("Brief dropout restarted collected frames");
            measured.lastGoodGrid = std::chrono::steady_clock::now() - std::chrono::seconds(2);
            measured.grid.clear();
            measured.updateGridStability();
            if (measured.stable) throw std::runtime_error("Expired grid history retained");
        }
        {
            App noisy;
            std::vector<Detection> fixture;
            for (int i = 0; i < 9; ++i) {
                Detection d;
                d.centre = { double(60 + i % 3 * 80), double(60 + i / 3 * 80) };
                d.area = 900;
                d.colour = i % 3;
                fixture.push_back(d);
            }
            auto run = [&]() {
                for (int t = 0; t < 30; ++t) {
                    noisy.grid = fixture;
                    for (auto& d : noisy.grid) d.centre.x += (t % 2 ? 1.8 : -1.8);
                    noisy.updateGridStability();
                }
                };
            noisy.noiseLimitPx = 1.5;
            run();
            if (noisy.stable != 20 || noisy.gridQualityReady)
                throw std::runtime_error("Noise rejection count/display incorrect");
            noisy.action('+');
            run();
            if (!noisy.gridQualityReady) throw std::runtime_error("Adjusted noise limit not applied");
            noisy.labelsOK = noisy.refsOK = noisy.planeOK = true;
            noisy.noiseLimitPx = 1.0;
            noisy.gridQualityReady = false;
            noisy.calibration();
            if (noisy.valid || noisy.message.find("noise spread") == std::string::npos)
                throw std::runtime_error("Fit failed to explain quality rejection");
        }
        {
            Mapping boundary;
            boundary.robot = nominalRobot();
            if (!nearBoundary(boundary, { 372.34, 68.30 }) || inRegion(boundary, { 372.34, 68.30 }))
                throw std::runtime_error("Edge display expanded allowed region");
            if (regionLabel(boundary, { 442.65, -7.8 }) != "INSIDE" || regionLabel(boundary, { 600, 200 }) != "OUTSIDE")
                throw std::runtime_error("Boundary classifications wrong");
        }
        // Exercise operator gates, real save/readback and independent-check recording.
        a = App{};
        a.opt.source = "demo";
        for (int i = 0; i < 25; ++i) {
            a.frame = synthetic(a);
            a.update();
        }
        a.calibration();
        if (a.valid) throw std::runtime_error("Operator confirmation gate bypassed");
        const auto original = fileutil::currentPath();
        fileutil::createDirectory("qa_outputs");
        fileutil::currentPath("qa_outputs");
        try {
            a.labelsOK = a.refsOK = a.planeOK = true;
            a.calibration();
            if (!a.valid) throw std::runtime_error("Calibration save failed");
            cv::FileStorage f((a.session + "/" + "calibration.yml"), cv::FileStorage::READ);
            cv::Mat stored;
            f["H_pixel_to_robot_xy"] >> stored;
            int physical = 1, sim = 0;
            f["physically_validated"] >> physical;
            f["simulation"] >> sim;
            if (physical != 0 || sim != 1 || cv::norm(stored - cv::Mat(a.mapping.h)) > 1e-8)
                throw std::runtime_error("Saved calibration metadata or matrix mismatch");
            a.test = true;
            a.compareNine();
            if (!a.comparisonView || !a.freeze || a.comparison.size() != 9 || !fileutil::exists(a.comparisonPath))
                throw std::runtime_error("Nine-point comparison export failed");
            for (int i = 0; i < 9; ++i)
                if (a.comparison[i].error > 2 || cv::norm(a.comparison[i].reference - a.refs[i]) > 1e-9)
                    throw std::runtime_error("Comparison point identity mismatch");
            a.action('x');
            if (a.freeze || a.comparisonView) throw std::runtime_error("Comparison did not return live");
            a.test = true;
            for (int i = 0; i < 15; ++i) {
                a.frame = synthetic(a);
                a.update();
            }
            a.beginMeasurement();
            if (!a.entering) throw std::runtime_error("Validation entry failed");
            a.input = "invalid";
            a.finishMeasurement();
            if (!a.entering) throw std::runtime_error("Invalid actual XY accepted");
            a.input = "427.65,-17.8";
            a.finishMeasurement();
            if (a.entering || !fileutil::exists(a.session + "/" + "independent_checks.csv"))
                throw std::runtime_error("Validation log failed");
            a.action('3');
            for (int i = 0; i < 25; ++i) {
                a.frame = synthetic(a);
                a.update();
            }
            for (int pattern = 1; pattern <= 3; ++pattern) {
                a.pattern = pattern;
                auto p = a.currentPlan();
                if (!p.valid) throw std::runtime_error("Three-colour planning failed");
                for (int l = 0; l < 3; ++l) {
                    if (p.colours[l] != PATTERNS[pattern - 1][2 - l]) throw std::runtime_error("Pattern order changed");
                    for (int j = 0; j < l; ++j)
                        if (p.source[l] == p.source[j]) throw std::runtime_error("Duplicate source in plan");
                }
            }
            a.action('p');
            if (!a.planLocked) throw std::runtime_error("Stable plan lock failed");
            a.savePlan();
            a.block.x += 8;
            a.frame = synthetic(a);
            a.update();
            if (a.planLocked || a.currentPlan().valid)
                throw std::runtime_error("Moved scene did not invalidate preview");
            auto absent = a.detections;
            absent.erase(std::remove_if(absent.begin(), absent.end(), [](auto& d) { return d.colour == 1; }),
                absent.end());
            if (makePlan(absent, a.mapping, 1).valid) throw std::runtime_error("Missing colour accepted");
            auto outside = a.detections;
            for (auto& d : outside)
                if (d.colour == 1) d.centre = project(a.mapping.h.inv(), { 600, 200 });
            if (makePlan(outside, a.mapping, 1).valid) throw std::runtime_error("Outside block planned");
            a.action('v');
            if (a.valid) throw std::runtime_error("Unconfirm did not invalidate");
            fileutil::currentPath(original);
        }
        catch (...) {
            fileutil::currentPath(original);
            throw;
        }
        std::cout
            << "PASS: automatic nine-top detection in 3 camera poses; 15 unseen points; worst synthetic XY error "
            << worst
            << " mm; operator gates, save/readback, validation logging, all 3 planning patterns, plan save, movement unlock, missing colour and outside-source rejection.\n";
        return true;
    }
    void mouse(int event, int x, int y, int, void* ptr) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            auto& a = *static_cast<App*>(ptr);
            try {
                a.click(x, y);
            }
            catch (const std::exception& e) {
                a.message = e.what();
            }
        }
    }
} // namespace calibrationui

// ------------------------------------------------------------
// SECTION 2  SHARED STATE
//    Constants, then the data the whole program shares: V13State (re-pick bookkeeping),
//    V10State (one dynamic cycle), StackVision (stack zone monitor) and App (everything else).
// ------------------------------------------------------------
constexpr int N = 9, CAM = 0, W = 1280, H = 1120, CAMW = 850, CAMH = 620;
constexpr int STABLE = 6, LOST_FRAMES = 20, ROI = 40;
constexpr double MIN_AREA = 350;
const string CONFIG = "handeye_v10_config.yml";
enum Colour { UNKNOWN = 0, RED = 1, GREEN = 2, BLUE = 3 };

const char* cname(int c) {
    switch (c) {
    case RED:
        return "RED";
    case GREEN:
        return "GREEN";
    case BLUE:
        return "BLUE";
    default:
        return "UNKNOWN";
    }
}
Scalar ink(int c) {
    switch (c) {
    case RED:
        return { 65, 85, 235 };
    case GREEN:
        return { 90, 210, 85 };
    case BLUE:
        return { 230, 155, 65 };
    default:
        return { 135, 145, 160 };
    }
}
const Scalar BG(24, 27, 35), PANEL(37, 42, 53), WHITE(235, 240, 245), MUTED(155, 167, 180), GOLD(55, 190, 245);

struct Position {
    int candidate = 0, streak = 0, confirmed = 0, missing = 0;
    bool uncertain = true;
};
struct Detection {
    Point centre;
    int colour;
    Rect box;
    double area;
    RotatedRect face;
    double fillRatio = 0;
    bool measurable = false;
};
struct BlockSample {
    Point centre;
    int colour;
    RotatedRect face;
    double area;
};
constexpr double BLOCK_MM = 35.0; // nominal measured cube edge; clearance bound remains separate
constexpr int SIZE_SAMPLES = 30;
struct Plan {
    bool valid = false;
    array<int, 3> source{ {-1, -1, -1} };
    array<int, 3> colours{ {0, 0, 0} };
    string message;
};

struct ReturnEvidence {
    std::deque<bool> samples;
    chrono::steady_clock::time_point lastSeen{}, started{}, lastUpdate{};
    double freshnessSeconds = .75;
    void clear() {
        samples.clear();
        lastSeen = {};
        started = {};
        lastUpdate = {};
        freshnessSeconds = .75;
    }
    void update(bool seen, chrono::steady_clock::time_point now) {
        if (!started.time_since_epoch().count()) started = now;
        double interval =
            lastUpdate.time_since_epoch().count() ? chrono::duration<double>(now - lastUpdate).count() : 0;
        // A gap invalidates old hits BEFORE adapting to the new acquisition cadence.
        // Thus a single image after a pause can never revive a completed confirmation.
        if (lastSeen.time_since_epoch().count() &&
            chrono::duration<double>(now - lastSeen).count() > freshnessSeconds) {
            samples.clear();
            lastSeen = {};
        }
        if (interval > 0 && interval <= 2.5)
            freshnessSeconds = max(.75, min(2.5, interval * 1.6 + .10));
        else if (interval > 2.5) {
            samples.clear();
            lastSeen = {};
            freshnessSeconds = .75;
        }
        lastUpdate = now;
        samples.push_back(seen);
        if (samples.size() > 12) samples.pop_front();
        if (seen) lastSeen = now;
    }
    bool ready(chrono::steady_clock::time_point now) const {
        return samples.size() >= 4 && std::count(samples.begin(), samples.end(), true) >= 4 &&
            lastSeen.time_since_epoch().count() &&
            chrono::duration<double>(now - lastSeen).count() <= freshnessSeconds;
    }
};
// User-measured open fingers at block height; opening direction confirmed robot Y.
constexpr double GRIPPER_OUTER_X_MM = 22.8, GRIPPER_OUTER_Y_MM = 71.0, GRIPPER_INNER_MM = 49.5;
constexpr double BLOCK_MAX_MM = 37.0, CLEARANCE_MARGIN_MM = 3.0;
// At12deg a37mm square projects43.88mm. ceil(max(blockProjection,(fingerSpan+blockProjection)/2)+3).
constexpr double MEASURED_GAP_X_MM = 47.0, MEASURED_GAP_Y_MM = 61.0;
// ---- Re-pick state
// Timing thresholds. Every transfer contains three 2 s gripper dwells, so no real transfer is
// shorter than V13_MIN_TRANSFER_S. A dwell inside the camera view lasts 2 s, so a scene that is
// still for V13_FALLBACK_STILL_S cannot be the arm pausing mid-transfer.
constexpr double V13_MIN_TRANSFER_S = 10.0;
constexpr double V13_SETTLE_S = 2.0;
constexpr double V13_ARMLESS_S = 20.0;
constexpr double V13_FALLBACK_STILL_S = 4.5;
constexpr double V13_FALLBACK_AFTER_S = 25.0;
constexpr double V13_EVIDENCE_S = 1.5;
constexpr double V13_SERIOUS_S = 6.0;
constexpr double V13_RESEND_S = 7.0;
constexpr int V13_MAX_RESENDS = 6;
constexpr int V13_MAX_RELEASE_RESENDS = 2;
constexpr double V13_LOST_ACCEPT_S = 8.0;
constexpr double V13_SOURCE_OVERRIDE_S = 6.0;
constexpr double V13_STILL_MM = 4.0;
constexpr double V13_SAME_PLACE_MM = 4.0;
constexpr double V13_NEW_BLOCK_MM = 20.0;
constexpr int V13_MAX_TRIES = 3;
constexpr double V13_NO_PROGRESS_S = 600.0;
constexpr double V13_NEAR_SOURCE_MM = 40.0; // a bump moves the missed block less than this from its source
constexpr double V13_UNDER_MM = 25.0;       // a block landed on another block covers that block's position
constexpr double V13_FROZEN_S = 1.5;        // bit-identical sparse samples for this long = frozen feed
struct SceneItem {
    int colour = 0;
    Point2d xy;
};
struct V13State {
    int attempt = 0;       // cycle-wide attempt id 1..127; MOD ignores a repeat
    array<int, 7> tries{}; // re-picks per checkpoint step 1..6
    int releasedStep = 0;  // checkpoint the robot is heading to after the last release
    chrono::steady_clock::time_point releasedAt{};
    bool armSeen = false; // workspace view was obstructed after the release
    chrono::steady_clock::time_point clearSince{}, stillSince{};
    vector<SceneItem> anchorScene, snapshot;
    int mismatch = 0;
    int pendingStep = 0, pendingMode = 0, pendingAttempt = 0, pendingTx = 0;
    Point2d pendingXY{};
    chrono::steady_clock::time_point pendingLastTx{};
    array<int, 7> retargetAttempt{};
    array<Point2d, 7> retargetXY{};
    array<int, 7> permitKind{}; // 0 none yet, 1 plain permit sent, 2 retarget sent: resends never change it
    int evidenceKind = 0;
    chrono::steady_clock::time_point evidenceSince{};
    deque<Point2d> samples;
    deque<chrono::steady_clock::time_point> sampleTimes;
    int sampleKey = -1;
    array<bool, 3> lost{ {false, false, false} };
    chrono::steady_clock::time_point lastProgress{}, sourceOverrideSince{}, nextSearchSince{};
    int repicks = 0, retargets = 0, accepted = 0;
    string hint; // overrides the checkpoint status for this tick
    Mat frameSig;
    chrono::steady_clock::time_point sameSince{};
    bool frozen = false;
    array<Mat, 7> stackBase;      // stack ROI when the robot left for checkpoint 1..6 (arm at HOME)
    array<bool, 7> deltaOK{};     // layer verified by the before/after stack comparison
    array<bool, 7> basePending{}; // baseline still to be taken on the next clear frame
    uint8_t releaseStart = 0, releaseEnd = 0;
    vector<uint8_t> releaseFields;
    int releaseResends = 0; // V13.5 frame that released this step
};
struct V10State {
    bool enlargedReturns = true, setupOpen = false, returnObserved = false, showClearance = true, allowRepeats = true;
    bool workspaceChecked = false;
    int phase = 0, token = 0, stableFrames = 0;
    bool recoveryArmed = false;
    double minX = 320, maxX = 540, minY = -160, maxY = 120;
    int nextPattern = 0;
    array<bool, 3> returnConfirmed{ {false, false, false} };
    double gapX = MEASURED_GAP_X_MM, gapY = MEASURED_GAP_Y_MM; // measured fingers + rotated37mm neighbour +3mm margin
    Mat emptyBackground;
    calibrationui::H backgroundMapping = calibrationui::H::eye();
    array<double, 4> backgroundRange{ {0, 0, 0, 0} };
    array<Point2d, 3> pick{}, drop{}, anchor{};
    array<int, 3> colours{};
    std::deque<array<Point2d, 3>> planHistory;
    array<ReturnEvidence, 3> returnEvidence{};
    std::deque<chrono::steady_clock::time_point> planTimes;
    int planOutliers = 0;
    array<Point2d, 3> pendingAnchor{};
    array<ReturnEvidence, 3> sourceEvidence{};
    array<bool, 3> currentReturns{ {false, false, false} };
    int handshakeStep = 1, operatorRequest = 0, retryControl = 0;
    bool resetArmed = false, resetAwait = false, resetPacketSent = false, pumpFailed = false;
    chrono::steady_clock::time_point controlLast{}, handshakeSince{};
    bool startRequested = false;
    chrono::steady_clock::time_point startDeadline{};
    string returnDetail;
    double visionPeriod = 0;
    chrono::steady_clock::time_point visionLast{};
    vector<Point2d> used;
    string status = "Calibrate and capture clear reference once, then place blocks and Start";
    chrono::steady_clock::time_point quietSince{}, lastReturnTx{}, cycleStarted{}, finishAfter{};
    V13State v13;
    bool autoRun = false;
    chrono::steady_clock::time_point completedAt{};
    Scalar exposureShift{ 0, 0, 0, 0 }; // current frame minus saved empty background, neutral pixels
};
// Visual evidence only: this cannot see or acknowledge an off-camera HOME pose.
struct StackVision {
    Rect workspace, stack;
    Mat reference;
    double faceArea = 900;
    bool savedReference = false;
    array<double, 4> captureBounds{ {0, 0, 0, 0} };
    array<double, 4> originalBounds{ {0, 0, 0, 0} };
    calibrationui::H captureMapping = calibrationui::H::eye();
    array<int, 3> observedLayers{ {-1, -1, -1} };
    int collectedTop = 0;
    Point2d collectedCentre;
    chrono::steady_clock::time_point collectedLast{}, collectedSince{};
    bool finalTopOnly = false;
    int finalFrames = 0;
    chrono::steady_clock::time_point finalSince{};
    bool configured = false, armed = false, clearStable = false, emptyStable = false, topClean = false;
    array<int, 3> expected{ {0, 1, 2} };
    int progress = 0, candidate = -2, streak = 0, clearFrames = 0, emptyFrames = 0, top = -1;
    int topCount = 0, currentFace = -1; // V13: number of square tops in the stack zone and the unique top colour
    calibrationui::DetectionFilter objectFilter;
    vector<Rect> referenceBlocks;
    chrono::steady_clock::time_point candidateSince{}, clearSince{}, emptySince{}, lastFrame{};
    Point2d candidateCentre;
    chrono::steady_clock::time_point candidateLastSeen{};
    string clearReason = "Collecting workspace evidence";
    Mat obstacleMask;
    string note = "Capture clear reference in Calibration";
    void resetFresh() {
        collectedTop = 0;
        collectedLast = {};
        collectedSince = {};
        finalFrames = 0;
        finalSince = {};
        clearStable = emptyStable = topClean = false;
        clearFrames = emptyFrames = streak = 0;
        candidate = -2;
        clearSince = {};
        emptySince = {};
        candidateSince = {};
        candidateLastSeen = {};
        lastFrame = {};
    }
    void begin(array<int, 3> order) {
        collectedTop = 0;
        collectedLast = {};
        collectedSince = {};
        observedLayers.fill(-1);
        finalFrames = 0;
        finalSince = {};
        expected = order;
        progress = 0;
        armed = true;
        candidate = -2;
        streak = 0;
        candidateSince = {};
        candidateLastSeen = {};
        note = "Watching stack colours bottom first";
    }
    bool fresh(chrono::steady_clock::time_point now) const {
        return configured && lastFrame.time_since_epoch().count() &&
            chrono::duration<double>(now - lastFrame).count() < .75;
    }
    bool startReady(chrono::steady_clock::time_point now) const { return fresh(now) && clearStable && emptyStable; }
    bool returnReady(chrono::steady_clock::time_point now) const {
        // NO STACK GUARD: the stack-surround / workspace-clear checks no longer gate the return.
        // Return is allowed as soon as every colour was seen in order (or, in FINAL TOP mode, the final colour was seen steadily).
        if (!fresh(now) || !armed || emptyStable)
            return false; // emptyStable: the stack is gone, nothing left to return
        return finalTopOnly ? finalFrames >= 8 && finalSince.time_since_epoch().count() &&
            chrono::duration<double>(now - finalSince).count() >= 1.0
            : progress == 3 && observedLayers == expected;
    }
    // Require a current final face for two seconds before sending the next plan.
    // Missing, wrong or ambiguous views restart this settling period.
    bool settledReturnReady(chrono::steady_clock::time_point now) const {
        return returnReady(now) && finalFrames >= 8 && finalSince.time_since_epoch().count() &&
            chrono::duration<double>(now - finalSince).count() >= 2.0;
    }
    bool finishReady(chrono::steady_clock::time_point now) const { return fresh(now) && clearStable && emptyStable; }
    static Rect padded(Rect r, int n, Size size) {
        return Rect(r.x - n, r.y - n, r.width + 2 * n, r.height + 2 * n) & Rect(0, 0, size.width, size.height);
    }
    void observe(const Mat& frame, const vector<calibrationui::Detection>& objects,
        chrono::steady_clock::time_point now) {
        if (!configured) return;
        if (frame.empty() || frame.size() != reference.size() ||
            (workspace & Rect(0, 0, frame.cols, frame.rows)) != workspace || (stack & workspace) != stack) {
            configured = false;
            resetFresh();
            note = "View changed: Teach Stack View again";
            return;
        }
        lastFrame = now;
        Mat delta;
        absdiff(reference, frame, delta);
        vector<Mat> channels;
        split(delta, channels);
        Mat changed;
        max(channels[0], channels[1], changed);
        max(changed, channels[2], changed);
        threshold(changed, changed, 40, 255, THRESH_BINARY);
        medianBlur(changed, changed, 3);
        Mat visible(frame.size(), CV_8U, Scalar(0));
        rectangle(visible, workspace, Scalar(255), FILLED);
        rectangle(visible, stack, Scalar(0), FILLED);
        for (auto r : referenceBlocks) rectangle(visible, padded(r, 8, frame.size()), Scalar(0), FILLED);
        vector<const calibrationui::Detection*> tops;
        for (const auto& d : objects) {
            double measuredArea = d.area, expectedArea = faceArea;
            if (savedReference && d.quad.size() == 4) {
                vector<Point2f> world;
                for (auto q : d.quad) {
                    auto w = calibrationui::project(captureMapping, q);
                    world.emplace_back(float(w.x), float(w.y));
                }
                measuredArea = abs(contourArea(world));
                expectedArea = BLOCK_MM * BLOCK_MM;
            }
            bool sizeOK =
                d.quad.size() == 4 && measuredArea >= expectedArea * .40 && measuredArea <= expectedArea * 2.5;
            if (sizeOK && (d.box & workspace).area() > d.box.area() * .8)
                rectangle(visible, padded(d.box, 8, frame.size()), Scalar(0), FILLED);
            if (sizeOK && (d.box & stack) == d.box) tops.push_back(&d);
        }
        // Robust exposure-offset correction from background pixels only. It cannot remove
        // a localized arm/obstruction; cap the permitted offset at 20 intensity levels.
        vector<int> offsets[3];
        for (int y = workspace.y; y < workspace.br().y; y += 4)
            for (int x = workspace.x; x < workspace.br().x; x += 4)
                if (visible.at<uchar>(y, x)) {
                    auto cur = frame.at<Vec3b>(y, x), ref = reference.at<Vec3b>(y, x);
                    for (int k = 0; k < 3; ++k) offsets[k].push_back(int(cur[k]) - int(ref[k]));
                }
        Scalar shift;
        for (int k = 0; k < 3; ++k) {
            auto& values = offsets[k];
            if (values.size() >= 50) {
                nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
                shift[k] = max(-20, min(20, values[values.size() / 2]));
            }
        }
        Mat adjusted;
        add(reference, shift, adjusted);
        absdiff(adjusted, frame, delta);
        split(delta, channels);
        max(channels[0], channels[1], changed);
        max(changed, channels[2], changed);
        threshold(changed, changed, 40, 255, THRESH_BINARY);
        medianBlur(changed, changed, 3);
        Mat hsv, red, redHigh, redChanged;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(0, 80, 45), Scalar(22, 255, 255), red);
        inRange(hsv, Scalar(168, 80, 45), Scalar(179, 255, 255), redHigh);
        red |= redHigh;
        bitwise_and(red, changed, redChanged);
        Mat blocked, redOutside;
        bitwise_and(changed, visible, blocked);
        bitwise_and(redChanged, visible, redOutside);
        int n = countNonZero(visible);
        int changedPixels = countNonZero(blocked), redPixels = countNonZero(redOutside);
        int changeLimit = max(max(100, int(faceArea * .5)), int(n * .01)), redLimit = max(20, int(faceArea * .05));
        bool enough = n >= max(200, int(workspace.area() * .12));
        bool clear =
            enough && double(changedPixels) / max(1, n) < .04 && changedPixels < changeLimit && redPixels < redLimit;
        obstacleMask = blocked.clone();
        clearReason = !enough ? "Too little visible background for workspace check"
            : redPixels >= redLimit ? "Red/orange change outside detected blocks: " + to_string(redPixels) +
            " px (limit " + to_string(redLimit) + ")"
            : !clear ? "Background differs from reference: " + to_string(changedPixels) + " px (limit " +
            to_string(changeLimit) + "); highlighted regions need inspection"
            : "Clear view: collecting " + to_string(clearFrames + 1) + "/5 frames and 0.5 seconds";
        if (clear) {
            if (!clearFrames) clearSince = now;
            ++clearFrames;
        }
        else {
            clearFrames = 0;
            clearSince = {};
        }
        clearStable = clearFrames >= 5 && chrono::duration<double>(now - clearSince).count() >= .5;
        bool empty = tops.empty() && double(countNonZero(changed(stack))) / max(1, stack.area()) < .04;
        if (empty) {
            if (!emptyFrames) emptySince = now;
            ++emptyFrames;
        }
        else {
            emptyFrames = 0;
            emptySince = {};
        }
        emptyStable = emptyFrames >= 5 && chrono::duration<double>(now - emptySince).count() >= .5;
        int layerColour = -1;
        bool layerView = false;
        top = -1;
        topClean = false;
        string topReason = tops.empty() ? "No fresh square top in stack zone"
            : "Multiple tops in stack zone: separate nearby blocks or reteach the zone";
        // ONE ordered colour history drives both the swatches and progress.
        // The unique fresh square face may be recorded while the wider stack area
        // is busy; complete stack/workspace clearance is required before RETURN.
        bool faceVisible = tops.size() == 1;
        if (faceVisible) layerColour = tops.front()->colour;
        topCount = int(tops.size());
        currentFace = faceVisible ? tops.front()->colour : -1;
        if (tops.size() == 1) {
            const auto& d = *tops[0];
            Mat ring = Mat::zeros(frame.size(), CV_8U);
            rectangle(ring, stack, Scalar(255), FILLED);
            int padding = max(10, int(sqrt(d.area) * .6));
            rectangle(ring, padded(d.box, padding, frame.size()), Scalar(0), FILLED);
            int guard = countNonZero(ring);
            Mat obstruction;
            bitwise_and(changed, ring, obstruction);
            Mat ringRed;
            bitwise_and(redChanged, ring, ringRed);
            topClean = guard >= max(150, int(stack.area() * .10)) &&
                double(countNonZero(obstruction)) / max(1, guard) < .12 &&
                countNonZero(ringRed) < max(20, int(faceArea * .05));
            // Per-layer colour needs a clear face and its immediate surroundings;
            // full stack/workspace clearance is independently required before return.
            Mat local = Mat::zeros(frame.size(), CV_8U);
            rectangle(local, padded(d.box, max(10, int(sqrt(d.area) * .5)), frame.size()) & stack, Scalar(255), FILLED);
            rectangle(local, padded(d.box, max(3, int(sqrt(d.area) * .1)), frame.size()), Scalar(0), FILLED);
            // Include dark lower faces and their attached shadow in the bounded stack body.
            // Colour alone omits these real parts of the stack. An oversized connected
            // foreground component is not masked: it must still block return.
            Mat paint = changed(stack).clone(), labels, stats, centres;
            morphologyEx(paint, paint, MORPH_CLOSE, getStructuringElement(MORPH_RECT, { 5, 5 }));
            int parts = connectedComponentsWithStats(paint, labels, stats, centres, 8, CV_32S);
            Point at(cvRound(d.centre.x) - stack.x, cvRound(d.centre.y) - stack.y);
            if (at.x >= 0 && at.y >= 0 && at.x < labels.cols && at.y < labels.rows) {
                int id = labels.at<int>(at);
                if (id > 0 && id < parts) {
                    Rect body(stats.at<int>(id, CC_STAT_LEFT) + stack.x, stats.at<int>(id, CC_STAT_TOP) + stack.y,
                        stats.at<int>(id, CC_STAT_WIDTH), stats.at<int>(id, CC_STAT_HEIGHT));
                    if (stats.at<int>(id, CC_STAT_AREA) <= d.area * 2.5 && body.width <= d.box.width * 1.8 &&
                        body.height <= d.box.height * 2.5) {
                        Rect footprint = padded(body, max(3, int(sqrt(d.area) * .15)), frame.size());
                        rectangle(local, footprint, Scalar(0), FILLED);
                        rectangle(ring, footprint, Scalar(0), FILLED);
                    }
                }
            }
            guard = countNonZero(ring);
            bitwise_and(changed, ring, obstruction);
            bitwise_and(redChanged, ring, ringRed);
            topClean = guard >= max(150, int(stack.area() * .10)) &&
                double(countNonZero(obstruction)) / max(1, guard) < .12 &&
                countNonZero(ringRed) < max(20, int(faceArea * .05));
            Mat localChange, localRed;
            bitwise_and(changed, local, localChange);
            bitwise_and(redChanged, local, localRed);
            int localN = countNonZero(local);
            layerView = localN >= 100 && double(countNonZero(localChange)) / max(1, localN) < .08 &&
                countNonZero(localRed) < max(15, int(d.area * .03));
            if (layerView) layerColour = d.colour;
            topClean = topClean && layerView;
            if (topClean) {
                top = d.colour;
                topReason = "Clean " + string(calibrationui::COLOUR_NAMES[top]) + " top";
            }
            else {
                topReason = countNonZero(ringRed) >= max(20, int(faceArea * .05))
                    ? "Top visible but red/orange obstruction in stack guard"
                    : "Top visible but surrounding stack guard is obstructed";
                topReason += " (outer changed " + to_string(countNonZero(obstruction)) + "/" + to_string(guard) +
                    ", red " + to_string(countNonZero(ringRed)) + "; near changed " +
                    to_string(countNonZero(localChange)) + "/" + to_string(localN) + ", red " +
                    to_string(countNonZero(localRed)) + ")";
            }
        }
        // Accumulate fresh clean final-top observations across short dropouts.
        // A paused sample is never sufficient to authorize return on an unclear frame.
        bool expired =
            collectedLast.time_since_epoch().count() && chrono::duration<double>(now - collectedLast).count() > .75;
        bool wrong = faceVisible && layerColour != expected[2];
        bool moved = faceVisible && collectedTop && norm(tops.front()->centre - collectedCentre) > 5.0;
        if (expired || wrong || moved) {
            collectedTop = 0;
            collectedLast = {};
            collectedSince = {};
        }
        if (armed && progress == 3 && topClean && clearStable && top == expected[2]) {
            if (!collectedTop) {
                collectedCentre = tops.front()->centre;
                collectedSince = now;
            }
            collectedTop = min(5, collectedTop + 1);
            collectedLast = now;
        }
        bool finalGood = armed && faceVisible && layerColour == expected[2] && !moved;
        if (finalGood) {
            if (!finalFrames) finalSince = now;
            ++finalFrames;
        }
        else {
            finalFrames = 0;
            finalSince = {};
        }
        // Occlusion is not a wrong colour: preserve a short window of verified,
        // clean observations. Never count the obscured image or infer skipped levels.
        if (!faceVisible) {
            bool keep = candidate >= 0 && candidateLastSeen.time_since_epoch().count() &&
                chrono::duration<double>(now - candidateLastSeen).count() <= 1.5 &&
                chrono::duration<double>(now - candidateSince).count() <= 3;
            if (!keep) {
                candidate = -2;
                streak = 0;
                candidateSince = {};
                candidateLastSeen = {};
            }
            note = emptyStable ? "Stack area empty"
                : keep ? topReason + "; retained clean samples " + to_string(streak)
                : topReason + "; no placement confirmed";
            return;
        }
        Point2d centre = tops.front()->centre;
        bool same = candidate == layerColour && candidateLastSeen.time_since_epoch().count() &&
            chrono::duration<double>(now - candidateLastSeen).count() <= 1.5 &&
            chrono::duration<double>(now - candidateSince).count() <= 3 &&
            norm(centre - candidateCentre) <= max(3.0, min(5.0, sqrt(tops.front()->area) * .10));
        if (!same) {
            candidate = layerColour;
            streak = 1;
            candidateSince = now;
            candidateCentre = centre;
        }
        else
            ++streak;
        candidateLastSeen = now;
        if (armed && progress < 3 && streak >= 2 && chrono::duration<double>(now - candidateSince).count() >= .05) {
            if (layerColour == expected[progress]) {
                observedLayers[progress] = layerColour;
                ++progress;
                candidate = -2;
                streak = 0;
                candidateSince = {};
                candidateLastSeen = {};
            }
        }
        if (armed && progress < 3) {
            // The previous layer on top is the NORMAL state between transfers, not an error.
            if (layerColour == expected[progress])
                note = "Stack " + to_string(progress) + "/3: next layer " + calibrationui::COLOUR_NAMES[layerColour] +
                " seen, confirming (" + to_string(min(streak, 2)) + "/2)";
            else if (progress > 0 && layerColour == expected[progress - 1])
                note = "Stack " + to_string(progress) + "/3: layer " + to_string(progress) + " " +
                calibrationui::COLOUR_NAMES[layerColour] + " on top; next " +
                calibrationui::COLOUR_NAMES[expected[progress]];
            else
                note = "Stack " + to_string(progress) + "/3: " + calibrationui::COLOUR_NAMES[layerColour] +
                " on top is OUT OF SEQUENCE (expected " + calibrationui::COLOUR_NAMES[expected[progress]] + ")";
        }
        else
            note = "3/3 colours matched; hold final top visible for 2s before return";
        if (finalTopOnly)
            note = string("FINAL TOP ONLY: ") +
            (top == expected[2] ? "expected top seen, samples " + to_string(finalFrames) + "/8 (1s minimum)"
                : "waiting for " + string(calibrationui::COLOUR_NAMES[expected[2]])) +
            "; lower colours NOT verified";
    }
};
struct App {
    bool finalTopFallback = false;
    int pendingControllerMode = 1;
    StackVision stackVision;
    array<StackVision, 4> visionProfiles;
    bool referenceDynamic = false;
    bool autoLightChecked = false, autoLightReady = false;
    int autoLightCandidate = -1, autoLightFrames = 0;
    array<double, 2> autoLightRaw{ {1e9, 1e9} }, autoLightComp{ {1e9, 1e9} }; // V13: last ON/OFF match scores
    chrono::steady_clock::time_point autoLightSince{}, autoLightLast{}, autoLightLoad{};
    int lightProfile = 0;
    bool lightingSetupOpen = false, lightCaptureActive = false, tuningOpen = false;
    int lightCaptureCount = 0;
    Mat lightCaptureSum, lightCapturePrevious, lightCaptureAnchor;
    calibrationui::H lightCaptureMapping = calibrationui::H::eye();
    chrono::steady_clock::time_point lightCaptureSince{};
    bool dynamicOperate = false;
    bool enlargedHover = false;
    double hoverMinX = 320, hoverMaxX = 550, hoverMinY = -276, hoverMaxY = 90;
    V10State v10;
    array<Point, N> points{};
    int mapped = 0;
    bool mapping = false;
    array<Position, N> positions{};
    int selected = 1;
    bool autoMode = true;
    bool busy = false;
    bool autoProposal = true;
    bool learnPick = false, learning = false;
    Point learnCentre{};
    int learnColour = UNKNOWN;
    vector<BlockSample> sizeSamples;
    Size2f learnedFace{};
    Size learnedResolution{};
    string sizeCapturePath;
    vector<Detection> lastDetections;
    Mat lastFrame;
    bool dynamicTestMode = false, dynamicSending = false;
    int dynamicStep = 0, dynamicToken = 0, dynamicColour = -1;
    chrono::steady_clock::time_point hoverSentAt{}, lastTestTx{};
    int buildCount = 0;          // blocks currently stacked at S1 by the TEST build flow
    array<Point2d, 3> buildXY{}; // where each stacked block was picked from (for RETURN ALL)
    chrono::steady_clock::time_point actionAt{};
    double actionSettle = 0;
    Point2d dynamicXY{};
    Point2d dynamicTrim{};
    string dynamicStatus = "Supervised test off";
    int dynamicRequest = 0;
    bool calibrationMode = false;
    calibrationui::App* calibration = nullptr;
    bool setupOpen = false;
    int pendingKey = -1;
    int completedCycles = 0;
    array<array<int, 3>, 3> sentColours{};
    array<chrono::steady_clock::time_point, 3> emptySince{}, retryAt{}, restoredSince{};
    array<bool, 3> returnStarted{};
    array<chrono::steady_clock::time_point, 3> stackSentAt{};
    bool sampleMode = false;
    int sampleColour = RED;
    array<int, 4> hue{ {0, 0, 60, 110} };
    array<int, 4> tolerance{ {0, 12, 15, 15} };
    int satMin = 65, valMin = 60;
    string message = "Load saved P1-P9 reference or calibrate once using M (9 blocks for auto proposal)";
    bool serialConnected = false;
    sp_port* port = nullptr;
    bool planReceiverReady = false;
    int lastSent = -1;
    string comName = "NONE";
    bool commandLatched = false;
    // V13.5: end of the last transmitted sequence (inter-frame gap) and the last motion frame sent.
    chrono::steady_clock::time_point txEnd{};
    uint8_t lastStart = 0, lastEnd = 0;
    vector<uint8_t> lastFields;
    bool planLocked = false;
    array<Plan, 3> queued{};
    bool allowRepeatPickups = false;  // default: unused blocks only
    array<bool, N> processedBlocks{}; // handled blocks tracked at their current return slots
    array<bool, N> reservedSources{}; // pickup positions used by a sent stack, freed by T
    chrono::steady_clock::time_point lastPlanRefresh{};
    bool planRefreshReady = false;
    array<bool, 3> sentPlans{ {false, false, false} }; // TX only, NOT ABB acknowledgement
    array<array<int, 3>, 3> sentSources{ {{{-1, -1, -1}}, {{-1, -1, -1}}, {{-1, -1, -1}}} };
    array<int, 3> sendOrder{ {0, 0, 0} };
    int sentCount = 0;
    // Random return plan (per stack): destination P index for level 1,2,3 (0-based).
    array<array<int, 3>, 3> returnDest{ {{{-1, -1, -1}}, {{-1, -1, -1}}, {{-1, -1, -1}}} };
    array<bool, 3> returnSent{ {false, false, false} };
    array<bool, N> returnReserved{}; // destinations promised to a pending return
    bool releaseArmed = false;       // second T forces release without a return plan
    mt19937 rng{ random_device{}() };
    bool autoCandidate = false;
    array<Point, N> candidatePoints{};
    int candidateStreak = 0;
    bool verifiedMap = false;
    bool showTargetGrid = true;
    int demoStep = 0;
    int zoneSetup = 0;
    array<Point, 3> stackZones{};
    array<bool, 3> zoneDefined{ {false, false, false} };
    array<int, 3> stackTop{ {0, 0, 0} }, topCandidate{ {0, 0, 0} }, topStreak{ {0, 0, 0} };
    array<vector<int>, 3> observedHistory{};
    Mat lastHSV;
    Size cameraSize{};
    Rect imageRect{ 15, 100, CAMW, CAMH };
};

// Small directory helpers keep this source compatible with Visual Studio's
// default C++14 mode; no std::filesystem or extra library is needed.

// ------------------------------------------------------------
// SECTION 3  SMALL HELPERS, SETTINGS FILE, STATIC MODE VISION AND PLANNING
// ------------------------------------------------------------
namespace fileutil {
    string currentPath() {
        char buffer[32768];
#ifdef _WIN32
        if (!_getcwd(buffer, sizeof(buffer))) throw runtime_error("cannot read working directory");
#else
        if (!getcwd(buffer, sizeof(buffer))) throw runtime_error("cannot read working directory");
#endif
        return buffer;
    }
    void currentPath(const string& path) {
#ifdef _WIN32
        if (_chdir(path.c_str()) != 0) throw runtime_error("cannot change working directory: " + path);
#else
        if (chdir(path.c_str()) != 0) throw runtime_error("cannot change working directory: " + path);
#endif
    }
    void createDirectory(const string& path) {
#ifdef _WIN32
        int result = _mkdir(path.c_str());
#else
        int result = mkdir(path.c_str(), 0777);
#endif
        if (result != 0) {
            struct stat info;
            if (errno != EEXIST || stat(path.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR))
                throw runtime_error("cannot create directory: " + path);
        }
    }
    bool exists(const string& path) {
        ifstream file(path, ios::binary);
        return file.good();
    }
}

// ---- Drawing helpers
void label(Mat& m, const string& s, Point p, double size = .56, Scalar c = WHITE, int thick = 1) {
    putText(m, s, p, FONT_HERSHEY_SIMPLEX, size, c, thick, LINE_AA);
}
void fill(Mat& m, Rect r, Scalar c) {
    rectangle(m, r, c, FILLED, LINE_AA);
}
void panel(Mat& m, Rect r) {
    fill(m, r, PANEL);
    rectangle(m, r, Scalar(57, 63, 76), 1);
}
void button(Mat& m, Rect r, const string& s, bool selected = false) {
    fill(m, r, selected ? Scalar(66, 93, 116) : Scalar(53, 60, 74));
    rectangle(m, r, selected ? GOLD : Scalar(77, 84, 97), 1);
    double font = .48;
    while (font > .24 && getTextSize(s, FONT_HERSHEY_SIMPLEX, font, 1, nullptr).width > r.width - 16) font -= .02;
    label(m, s, Point(r.x + 8, r.y + r.height / 2 + 5), font, selected ? WHITE : MUTED, 1);
}
// Camera/UI pump remains on the main thread while a packet sender owns the port.
static std::function<void()> serialVisionPump;

void resetObservations(App& a) {
    a.positions.fill(Position{});
}

// ---- Settings file
bool saveConfig(const App& a) {
    FileStorage f(CONFIG, FileStorage::WRITE);
    if (!f.isOpened()) return false;
    f << "dynamicTrimX" << a.dynamicTrim.x << "dynamicTrimY" << a.dynamicTrim.y;
    f << "allowRepeatPickups" << int(a.allowRepeatPickups);
    f << "processedBlocks" << "[";
    for (bool used : a.processedBlocks) f << int(used);
    f << "]";
    f << "autoProposal" << int(a.autoProposal);
    f << "learnedWidthPx" << a.learnedFace.width << "learnedHeightPx" << a.learnedFace.height;
    f << "learnedResolution" << a.learnedResolution << "sizeCapturePath" << a.sizeCapturePath;
    f << "verifiedMap" << int(a.verifiedMap) << "mapped" << a.mapped << "points" << "[";
    for (int i = 0; i < a.mapped; ++i) f << a.points[i];
    f << "]";
    for (int i = 0; i < 3; ++i) {
        f << ("zone" + to_string(i)) << a.stackZones[i];
        f << ("zoneDefined" + to_string(i)) << int(a.zoneDefined[i]);
    }
    f << "satMin" << a.satMin << "valMin" << a.valMin;
    for (int c = 1; c <= 3; ++c) {
        f << ("hue" + to_string(c)) << a.hue[c];
        f << ("tol" + to_string(c)) << a.tolerance[c];
    }
    return true;
}
void loadConfig(App& a) {
    FileStorage f(CONFIG, FileStorage::READ);
    if (!f.isOpened()) return;
    if (!f["autoProposal"].empty()) {
        int v;
        f["autoProposal"] >> v;
        a.autoProposal = v != 0;
    }
    if (!f["learnedWidthPx"].empty()) {
        f["learnedWidthPx"] >> a.learnedFace.width;
        f["learnedHeightPx"] >> a.learnedFace.height;
        f["learnedResolution"] >> a.learnedResolution;
        f["sizeCapturePath"] >> a.sizeCapturePath;
    }
    if (!f["dynamicTrimX"].empty()) {
        f["dynamicTrimX"] >> a.dynamicTrim.x;
        f["dynamicTrimY"] >> a.dynamicTrim.y;
        if (!isfinite(a.dynamicTrim.x) || !isfinite(a.dynamicTrim.y) || abs(a.dynamicTrim.x) > 5 ||
            abs(a.dynamicTrim.y) > 5)
            a.dynamicTrim = {};
    }
    if (!f["allowRepeatPickups"].empty()) {
        int enabled = 0;
        f["allowRepeatPickups"] >> enabled;
        a.allowRepeatPickups = enabled != 0;
    }
    int historyIndex = 0;
    for (auto entry : f["processedBlocks"]) {
        int used = 0;
        entry >> used;
        if (historyIndex < N) a.processedBlocks[historyIndex++] = used != 0;
    }
    int n = 0;
    f["mapped"] >> n;
    FileNode p = f["points"];
    int i = 0;
    for (auto it = p.begin(); it != p.end() && i < N; ++it) {
        (*it) >> a.points[i++];
    }
    a.mapped = (n == N && i == N) ? N : 0;
    int verified = 0;
    if (!f["verifiedMap"].empty()) f["verifiedMap"] >> verified;
    a.verifiedMap = (a.mapped == N && verified == 1);
    for (int j = 0; j < 3; ++j) {
        int defined = 0;
        auto zn = f["zone" + to_string(j)], zd = f["zoneDefined" + to_string(j)];
        if (!zn.empty()) zn >> a.stackZones[j];
        if (!zd.empty()) zd >> defined;
        a.zoneDefined[j] = (defined == 1);
    }
    if (!f["satMin"].empty()) f["satMin"] >> a.satMin;
    if (!f["valMin"].empty()) f["valMin"] >> a.valMin;
    for (int c = 1; c <= 3; ++c) {
        auto h = f["hue" + to_string(c)], t = f["tol" + to_string(c)];
        if (!h.empty()) h >> a.hue[c];
        if (!t.empty()) t >> a.tolerance[c];
    }
}

// ---- Static mode vision (colour sampling, block capture, grid proposal)
void findColour(const Mat& hsv, Mat& camera, int colour, const App& a, vector<Detection>& detections) {
    calibrationui::ColourModel cfg;
    cfg.staticStyle = true;
    cfg.sat = a.satMin;
    cfg.val = a.valMin;
    for (int k = 0; k < 3; ++k) {
        cfg.hue[k] = a.hue[k + 1];
        cfg.tol[k] = a.tolerance[k + 1];
    }
    Mat mask = calibrationui::colourThreshold(hsv, colour - 1, cfg);
    calibrationui::cleanColourMask(mask, true);
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < MIN_AREA) continue;
        Moments mo = moments(contour);
        if (mo.m00 == 0) continue;
        Point p(cvRound(mo.m10 / mo.m00), cvRound(mo.m01 / mo.m00));
        Rect box = boundingRect(contour);
        RotatedRect face = minAreaRect(contour);
        double shortSide = min(face.size.width, face.size.height);
        double longSide = max(face.size.width, face.size.height);
        double ratio = area / max(1.0, double(face.size.area()));
        bool measurable = shortSide >= 12 && longSide / max(1.0, shortSide) <= 1.45 && ratio >= .80 && box.x > 2 &&
            box.y > 2 && box.br().x < hsv.cols - 2 && box.br().y < hsv.rows - 2;
        detections.push_back({ p, colour, box, area, face, ratio, measurable });
        Point2f corners[4];
        face.points(corners);
        for (int i = 0; i < 4; ++i) line(camera, corners[i], corners[(i + 1) % 4], ink(colour), 2, LINE_AA);
        string text =
            string(cname(colour)) + " " + to_string(cvRound(shortSide)) + "x" + to_string(cvRound(longSide)) + "px";
        label(camera, text, Point(box.x, max(17, box.y - 5)), .43, ink(colour), 1);
    }
}

// Reset is persisted; automatic grid proposal stays off until the operator maps again.
void resetSettings(App& a, bool all) {
    if (a.busy || a.sentCount || a.v10.phase) {
        a.message = "Reset unavailable during a cycle";
        return;
    }
    if (all) {
        auto calibration = a.calibration;
        auto port = a.port;
        bool connected = a.serialConnected;
        string com = a.comName;
        int last = a.lastSent, cycles = a.completedCycles;
        bool latched = a.commandLatched;
        a = App{};
        a.calibration = calibration;
        a.port = port;
        a.serialConnected = connected;
        a.comName = com;
        a.lastSent = last;
        a.completedCycles = cycles;
        a.commandLatched = latched;
    }
    a.processedBlocks.fill(false); // mapping IDs are no longer valid after remapping
    a.points = {};
    a.mapped = 0;
    a.mapping = false;
    a.verifiedMap = false;
    a.zoneDefined.fill(false);
    a.stackZones = {};
    a.stackTop.fill(0);
    a.topCandidate.fill(0);
    a.topStreak.fill(0);
    for (auto& h : a.observedHistory) h.clear();
    a.autoProposal = false;
    a.autoCandidate = false;
    a.candidateStreak = 0;
    a.sampleMode = false;
    a.zoneSetup = 0;
    a.learnPick = false;
    a.learning = false;
    a.sizeSamples.clear();
    a.queued = {};
    a.planRefreshReady = false;
    a.planLocked = false;
    resetObservations(a);
    a.setupOpen = true;
    if (!saveConfig(a)) {
        a.message = "Reset applied in memory but settings file cannot be written";
        return;
    }
    a.message = all ? "Settings reset and saved. COM kept open; measurement archives retained."
        : "P1-P9 and stack zones cleared and saved. Click MAP to start again.";
}

float medianValue(vector<float> v) {
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}
void saveBlockCapture(App& a) {
    try {
        string id = to_string(
            chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count());
        fileutil::createDirectory("block_measurements");
        string folder = "block_measurements/" + id;
        fileutil::createDirectory(folder);
        ofstream csv(folder + "/samples.csv");
        if (!csv) throw runtime_error("cannot write samples.csv");
        csv << "sample,colour,centre_u_px,centre_v_px,rect_width_px,rect_height_px,angle_deg,contour_area_px2,corner0_u,corner0_v,corner1_u,corner1_v,corner2_u,corner2_v,corner3_u,corner3_v,known_face_mm\n";
        vector<float> widths, heights;
        for (size_t i = 0; i < a.sizeSamples.size(); ++i) {
            const auto& r = a.sizeSamples[i];
            widths.push_back(min(r.face.size.width, r.face.size.height));
            heights.push_back(max(r.face.size.width, r.face.size.height));
            csv << i << ',' << cname(r.colour) << ',' << r.centre.x << ',' << r.centre.y << ',' << r.face.size.width
                << ',' << r.face.size.height << ',' << r.face.angle << ',' << r.area;
            Point2f corners[4];
            r.face.points(corners);
            for (const auto& p : corners) csv << ',' << p.x << ',' << p.y;
            csv << ',' << BLOCK_MM << '\n';
        }
        csv.flush();
        if (!csv) throw runtime_error("sample write failed");
        Size2f learned(medianValue(widths), medianValue(heights));
        FileStorage meta(folder + "/measurement.yml", FileStorage::WRITE);
        if (!meta.isOpened()) throw runtime_error("cannot write measurement.yml");
        meta
            << "known_face_mm" << BLOCK_MM << "sample_count" << int(a.sizeSamples.size()) << "camera_index" << CAM
            << "frame_size" << a.cameraSize << "median_short_side_px" << learned.width << "median_long_side_px"
            << learned.height << "local_mm_per_px_short" << BLOCK_MM / learned.width << "local_mm_per_px_long"
            << BLOCK_MM / learned.height << "scope"
            << "Local apparent face size only; not robot XY calibration. Same camera pose, resolution and block height required.";
        meta.release();
        if (!imwrite(folder + "/camera.png", a.lastFrame)) throw runtime_error("snapshot write failed");
        ofstream visible(folder + "/visible_blocks.csv");
        if (!visible) throw runtime_error("cannot write visible_blocks.csv");
        visible << "colour,centre_u_px,centre_v_px,width_px,height_px,angle_deg,area_px2,measurable\n";
        for (const auto& d : a.lastDetections)
            visible << cname(d.colour) << ',' << d.centre.x << ',' << d.centre.y << ',' << d.face.size.width << ','
            << d.face.size.height << ',' << d.face.angle << ',' << d.area << ',' << d.measurable << '\n';
        visible.flush();
        if (!visible) throw runtime_error("visible block export failed");
        a.learnedFace = learned;
        a.learnedResolution = a.cameraSize;
        a.sizeCapturePath = folder;
        if (!saveConfig(a)) throw runtime_error("cannot persist reference in handeye_config.yml");
        a.message = "40mm reference saved: " + to_string(cvRound(learned.width)) + "x" +
            to_string(cvRound(learned.height)) + " camera px; " + folder;
    }
    catch (const exception& e) {
        a.message = string("Measurement save failed: ") + e.what();
    }
    a.learning = false;
}
void updateBlockCapture(App& a, const vector<Detection>& det) {
    if (!a.learning) return;
    const Detection* closest = nullptr;
    double best = 24.0;
    for (const auto& d : det)
        if (d.colour == a.learnColour && d.measurable) {
            double distance = norm(d.centre - a.learnCentre);
            if (distance < best) {
                closest = &d;
                best = distance;
            }
        }
    if (!closest) {
        a.sizeSamples.clear();
        a.message = "LEARN: block lost / face unclear. Hold still or click Cancel learn.";
        return;
    }
    if (!a.sizeSamples.empty()) {
        const auto& first = a.sizeSamples.front();
        double w = min(closest->face.size.width, closest->face.size.height);
        double h = max(closest->face.size.width, closest->face.size.height);
        double fw = min(first.face.size.width, first.face.size.height),
            fh = max(first.face.size.width, first.face.size.height);
        if (norm(closest->centre - first.centre) > 4 || abs(w - fw) > fw * .08 || abs(h - fh) > fh * .08)
            a.sizeSamples.clear();
    }
    a.sizeSamples.push_back({ closest->centre, closest->colour, closest->face, closest->area });
    a.learnCentre = closest->centre;
    a.message = "LEARNING 40x40mm face: " + to_string(a.sizeSamples.size()) + "/30 stable frames - keep block still";
    if (a.sizeSamples.size() >= SIZE_SAMPLES) saveBlockCapture(a);
}

bool proposeGrid(const vector<Detection>& det, array<Point, N>& out) {
    if (det.size() != N) return false;
    vector<Point> p;
    for (const auto& d : det) p.push_back(d.centre);
    sort(p.begin(), p.end(), [](Point a, Point b) { return a.y < b.y; });
    for (int row = 0; row < 3; ++row) {
        sort(p.begin() + row * 3, p.begin() + row * 3 + 3, [](Point a, Point b) { return a.x < b.x; });
        for (int col = 0; col < 3; ++col) out[row * 3 + col] = p[row * 3 + col];
    }
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if (norm(out[i] - out[j]) < ROI * 1.2) return false;
    double y0 = 0, y1 = 0, y2 = 0;
    for (int j = 0; j < 3; ++j) {
        y0 += out[j].y;
        y1 += out[3 + j].y;
        y2 += out[6 + j].y;
    }
    return y1 - y0 > ROI * 1.5 && y2 - y1 > ROI * 1.5;
}

array<int, N> matchDetections(const App& a, const vector<Detection>& det) {
    array<int, N> assignment;
    assignment.fill(-1);
    vector<bool> used(det.size(), false);
    for (int pass = 0; pass < N; ++pass) {
        int bestI = -1, bestJ = -1;
        double best = double(ROI) * ROI;
        for (int i = 0; i < N; ++i)
            if (assignment[i] < 0)
                for (int j = 0; j < (int)det.size(); ++j)
                    if (!used[j]) {
                        Point delta = det[j].centre - a.points[i];
                        double d = delta.dot(delta);
                        if (d < best) {
                            best = d;
                            bestI = i;
                            bestJ = j;
                        }
                    }
        if (bestI < 0) break;
        assignment[bestI] = bestJ;
        used[bestJ] = true;
    }
    return assignment;
}

// ---- Static mode planning
Plan makePlan(const App& a, int routine) {
    Plan p; // target TOP -> BOTTOM
    const array<array<int, 3>, 3> top{ {{{GREEN, RED, BLUE}}, {{RED, GREEN, BLUE}}, {{BLUE, GREEN, RED}}} };
    for (int level = 0; level < 3; ++level) {
        int colour = top[routine - 1][2 - level];
        p.colours[level] = colour;
        for (int j = 0; j < N; ++j) {
            bool used = false;
            for (int k = 0; k < level; ++k)
                if (p.source[k] == j) used = true;
            if (!used && !a.reservedSources[j] && (a.allowRepeatPickups || !a.processedBlocks[j]) &&
                !a.positions[j].uncertain && a.positions[j].confirmed == colour) {
                p.source[level] = j;
                break;
            }
        }
        if (p.source[level] < 0) {
            p.message =
                string(a.allowRepeatPickups ? "Need stable " : "Need unused stable ") + cname(colour) + " block";
            return p;
        }
    }
    p.valid = true;
    p.message = "Ready (vision plan only)";
    return p;
}
void refreshPlans(App& a) {
    for (int s = 1; s <= 3; ++s) a.queued[s - 1] = makePlan(a, s);
    a.lastPlanRefresh = chrono::steady_clock::now();
    a.planRefreshReady = true;
}

// ------------------------------------------------------------
// SECTION 4  SERIAL LINK
//    Writes one value every 250 ms on a background thread while the camera keeps running.
//    See the protocol table at the top of the file. The robot never answers; the camera confirms.
// ------------------------------------------------------------
string binary8(int value) {
    return bitset<8>(static_cast<unsigned long long>(value & 255)).to_string();
}
string byteDescription(int value) {
    ostringstream os;
    os << "DEC " << value << "  HEX 0x" << uppercase << hex << setw(2) << setfill('0') << value << "  BIN "
        << binary8(value);
    if (value >= 226 && value <= 229) {
        const char* names[] = { "STATIC", "DYNAMIC", "TEST", "CALIBRATE" };
        os << "  MODE " << names[value - 226];
    }
    return os.str();
}
bool sendCommand(App& a, uint8_t value) {
    if (a.calibrationMode && !a.dynamicSending) {
        a.message = "Serial commands disabled in calibration/test mode";
        return false;
    }
    if (!a.port || !a.serialConnected) {
        a.message = "SERIAL OFFLINE: launch with COM port argument";
        return false;
    }
    int written = sp_blocking_write(a.port, &value, 1, 1000);
    if (written != 1) {
        a.serialConnected = false;
        a.message = "ERROR: serial write failed for " + to_string((int)value);
        cerr << a.message << endl;
        return false;
    }
    a.lastSent = (int)value;
    a.message = "TX: " + byteDescription(value) + " (write only; no ABB ACK)";
    cout << a.message << endl;
    return true;
}
constexpr int SERIAL_SYMBOL_MS = 250;
// V13.5: idle zero held between two frames. After an end byte the controller prints to the pendant and
// loops back to its listener; a start byte that arrives before it listens again is lost (one-way link).
constexpr int SERIAL_FRAME_GAP_MS = 1500;
vector<uint8_t> serialFrameSequence(uint8_t start, const vector<uint8_t>& fields, uint8_t end) {
    vector<uint8_t> sequence{ 0, start };
    for (auto field : fields) {
        sequence.push_back(0);
        sequence.push_back(field);
    }
    sequence.push_back(0);
    sequence.push_back(end);
    sequence.push_back(0);
    return sequence;
}
void writeTimedSymbols(const vector<uint8_t>& sequence, int symbolMs, const function<bool(uint8_t)>& write,
    atomic<size_t>& progress, atomic<int>& failure, atomic<bool>& done) {
    for (size_t i = 0; i < sequence.size(); ++i) {
        auto sentAt = chrono::steady_clock::now();
        if (!write(sequence[i])) {
            failure.store(int(i));
            break;
        }
        progress.store(i + 1);
        if (i + 1 < sequence.size()) this_thread::sleep_until(sentAt + chrono::milliseconds(symbolMs));
    }
    done.store(true);
}
// Self-tests record complete frames here instead of writing to a COM port.
static vector<vector<uint8_t>>* serialTestCapture = nullptr;
// Sends one complete sequence (frame) at 250 ms per value and keeps the camera running meanwhile.
bool sendTimedSequence(App& a, const vector<uint8_t>& sequence) {
    if (serialTestCapture) {
        serialTestCapture->push_back(sequence);
        if (sequence.size() > 1) a.lastSent = sequence[sequence.size() - 2];
        return true;
    }
    if (!a.port || !a.serialConnected) {
        a.message = "SERIAL OFFLINE: launch with COM port argument";
        return false;
    }
    if (a.txEnd.time_since_epoch().count()) {
        auto ready = a.txEnd + chrono::milliseconds(SERIAL_FRAME_GAP_MS);
        while (chrono::steady_clock::now() < ready) {
            a.message = "Short pause before the next command";
            if (serialVisionPump) serialVisionPump();
            waitKey(1);
        }
    }
    auto* port = a.port;
    atomic<size_t> progress{ 0 };
    atomic<int> failure{ -1 };
    atomic<bool> done{ false };
    thread sender([&]() {
        writeTimedSymbols(
            sequence, SERIAL_SYMBOL_MS, [&](uint8_t value) { return sp_blocking_write(port, &value, 1, 1000) == 1; },
            progress, failure, done);
        });
    size_t reported = 0;
    auto report = [&]() {
        size_t sent = progress.load();
        if (sent > reported) {
            reported = sent;
            a.lastSent = sequence[sent - 1];
            a.message = "Sending to the robot (" + to_string(sent) + "/" + to_string(sequence.size()) + ")";
        }
        };
    try {
        while (!done.load()) {
            report();
            if (serialVisionPump) serialVisionPump();
            waitKey(1);
        }
    }
    catch (...) {
        sender.join();
        throw;
    }
    sender.join();
    report();
    a.txEnd = chrono::steady_clock::now();
    if (failure.load() >= 0) {
        a.serialConnected = false;
        a.message = "Serial write failed; controller state unknown. Inspect pendant.";
        return false;
    }
    a.message = "Sent. The camera will confirm what the robot did.";
    return true;
}
void disconnect(App& a) {
    if (a.port) {
        sp_close(a.port);
        sp_free_port(a.port);
        a.port = nullptr;
    }
    a.serialConnected = false;
    a.comName = "NONE";
}
void startSerial(App& a, const string& name) {
    sp_port* p = nullptr;
    if (sp_get_port_by_name(name.c_str(), &p) != SP_OK || !p) {
        a.message = "Serial port not found: " + name;
        return;
    }
    if (sp_open(p, SP_MODE_WRITE) != SP_OK) {
        sp_free_port(p);
        a.message = "Cannot open serial port: " + name;
        return;
    }
    bool ok = sp_set_baudrate(p, 9600) == SP_OK && sp_set_bits(p, 8) == SP_OK &&
        sp_set_parity(p, SP_PARITY_NONE) == SP_OK && sp_set_stopbits(p, 1) == SP_OK &&
        sp_set_flowcontrol(p, SP_FLOWCONTROL_NONE) == SP_OK;
    if (!ok) {
        sp_close(p);
        sp_free_port(p);
        a.message = "Serial configuration failed";
        return;
    }
    a.port = p;
    a.serialConnected = true;
    a.comName = name;
    if (!sendCommand(a, 0)) {
        disconnect(a);
        return;
    }
    a.message = "COM OPEN " + name + " 9600 8N1; initial 00000000 sent";
}

// STOP/CLEAR sends 0. It does NOT halt RAPID or robot motion.
void operatorClear(App& a) {
    if (a.calibrationMode) return;
    if (a.busy) return;
    if (sendCommand(a, 0)) {
        a.commandLatched = false;
        a.message = "CLEAR 00000000 SENT; NOT A ROBOT STOP / NO ACK";
    }
}

// Send a framed list of bytes, each separated by 0 and held for the RAPID scheduler.
// Non-motion mode byte: 226 STATIC,227 DYNAMIC,228 TEST,229 CALIBRATE.
bool sendControllerMode(App& a, int mode) {
    if (mode < 1 || mode > 4 || !a.port || !a.serialConnected) return false;
    struct ModeGuard {
        App& app;
        bool busy, sending;
        ModeGuard(App& x) : app(x), busy(x.busy), sending(x.dynamicSending) {
            app.busy = true;
            app.dynamicSending = true;
        }
        ~ModeGuard() {
            app.busy = busy;
            app.dynamicSending = sending;
        }
    } guard(a);
    bool ok = sendTimedSequence(a, { 0, uint8_t(225 + mode), 0 });
    if (ok) {
        a.message = "MODE requested: " +
            string(mode == 1 ? "STATIC"
                : mode == 2 ? "DYNAMICv2"
                : mode == 3 ? "TEST"
                : "CALIBRATE") +
            "; no ACK - check pendant";
        cout << a.message << endl;
    }
    return ok;
}
// Frame layout: 0, START, 0, f0, 0, f1, ... 0, END, 0
static bool sendFrame(App& a, uint8_t start, const vector<uint8_t>& fields, uint8_t end) {
    if (a.busy) return false;
    if (a.calibrationMode && !a.dynamicSending) return false;
    // Reassert the mode before each new motion session in case a UI-only request was missed.
    int mode = start == 240 ? 1 : start == 220 ? 2 : start == 232 && fields.size() > 1 && fields[1] == 1 ? 3 : 0;
    if (mode && !sendControllerMode(a, mode)) return false;
    struct Guard {
        bool& b;
        Guard(bool& v) : b(v) { b = true; }
        ~Guard() { b = false; }
    } guard(a.busy);
    bool ok = sendTimedSequence(a, serialFrameSequence(start, fields, end));
    if (ok) {
        a.lastStart = start;
        a.lastFields = fields;
        a.lastEnd = end;
    }
    return ok;
}

// ------------------------------------------------------------
// SECTION 5  STACK MONITOR AND AUTOMATIC LIGHTING
//    The stack zone monitor reports which colours are on the stack and whether the workspace is clear.
//    Lighting: an empty background is saved for lamp ON and lamp OFF. The live view is compared tile by
//    tile with both, and the closer one is selected automatically (small exposure drift is allowed).
// ------------------------------------------------------------
calibrationui::ColourModel stackColourModel(const App& a) {
    calibrationui::ColourModel c;
    c.staticStyle = true;
    c.sat = a.satMin;
    c.val = a.valMin;
    c.minArea = int(MIN_AREA);
    for (int k = 0; k < 3; ++k) {
        c.hue[k] = a.hue[k + 1];
        c.tol[k] = a.tolerance[k + 1];
    }
    return c;
}
// Largest lighting-reference difference still accepted as a LIGHT ON / LIGHT OFF match (was 12).
// Adjustable on the TUNING page (default 15).
inline double lightMatchLimit() { return calibrationui::detectionTuning().matchLimit; }
constexpr double TUNE_MATCH_MIN = 5, TUNE_MATCH_MAX = 40;
constexpr int TUNE_SAT_MIN = 15, TUNE_SAT_MAX = 150, TUNE_VAL_MIN = 10, TUNE_VAL_MAX = 120;
void drawTuningPage(Mat& ui, App& a);
bool tuningMouse(App& a, int x, int y);
string stackStartReason(const App& a) {
    if (a.autoLightChecked && !a.autoLightReady) {
        auto sc = [](double x) { return x > 1e8 ? string("n/a") : calibrationui::fmt(x, 1); };
        return "Lighting: waiting for a stable match (ON " + sc(a.autoLightComp[0]) + ", OFF " +
            sc(a.autoLightComp[1]) + "; need <= " + calibrationui::fmt(lightMatchLimit(), 0) + "). n/a = reference missing or made with another calibration/area";
    }
    const auto& v = a.stackVision;
    if (!v.configured)
        return string(a.referenceDynamic ? "DYNAMIC" : "STATIC") + " " +
        (a.lightProfile == 0 ? "LIGHT ON" : "LIGHT OFF") +
        " reference missing: open Light / Background and capture the empty workspace";
    auto now = chrono::steady_clock::now();
    if (!v.fresh(now)) {
        double age = v.lastFrame.time_since_epoch().count() ? chrono::duration<double>(now - v.lastFrame).count() : -1;
        return "Waiting for fresh workspace check (age " + calibrationui::fmt(age, 2) + " s); " + v.note;
    }
    if (!v.clearStable) return "Workspace check: " + v.clearReason;
    if (!v.emptyStable)
        return "Stack check: waiting for empty view (" + to_string(v.emptyFrames) + " frames); " + v.note;
    return "";
}
void tickAutomaticLighting(App& a, const Mat& frame,
    chrono::steady_clock::time_point now = chrono::steady_clock::now());
void updateStackVision(App& a, const Mat& frame) {
    tickAutomaticLighting(a, frame);
    a.stackVision.finalTopOnly = a.finalTopFallback;
    auto& ref = a.stackVision;
    if (ref.savedReference &&
        (!a.calibration || !a.calibration->valid || frame.size() != ref.reference.size() ||
            norm(Mat(ref.captureMapping) - Mat(a.calibration->mapping.h)) > 1e-8 ||
            ref.captureBounds != array<double, 4>{{a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY}} ||
            ref.originalBounds != array<double, 4>{{a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY}})) {
        ref.configured = false;
        ref.resetFresh();
        a.v10.emptyBackground.release();
        a.v10.workspaceChecked = false;
        ref.note = "Calibration/area changed: capture clear reference in CALIBRATE";
        return;
    }
    if (a.dynamicOperate && a.calibration && a.calibration->freeze) {
        a.stackVision.resetFresh();
        return;
    }
    vector<calibrationui::Detection> objects;
    if (a.dynamicOperate && a.calibration && a.calibration->test && a.calibration->useStaticDetector)
        objects = a.calibration->detections;
    else {
        auto model = stackColourModel(a);
        auto& tracker = a.stackVision.objectFilter;
        objects = tracker.reacquire(frame, model, calibrationui::detect(frame, model));
        tracker.update(objects);
    }
    a.stackVision.observe(frame, objects, chrono::steady_clock::now());
}
int visionProfileSlot(const App& a) {
    return (a.referenceDynamic ? 2 : 0) + a.lightProfile;
}
string lightName(const App& a) {
    return a.lightProfile == 0 ? "LIGHT ON" : "LIGHT OFF";
}
string visionStem(const App& a) {
    return string(a.referenceDynamic ? "dynamic" : "static") + "_stack_vision_" +
        (a.lightProfile == 0 ? "light_on" : "light_off");
}
string visionSettings(const App& a) {
    return visionStem(a) + ".yml";
}
string visionImage(const App& a) {
    return visionStem(a) + "_reference.png";
}
void loadLightingPreference(App& a) {
    if (!fileutil::exists("workspace_lighting.yml")) return;
    try {
        FileStorage f("workspace_lighting.yml", FileStorage::READ);
        int selected = -1;
        f["selected_profile"] >> selected;
        if (selected == 0 || selected == 1) a.lightProfile = selected;
        calibrationui::detectionTuning().active = a.lightProfile;
    }
    catch (...) {}
}
void saveLightingPreference(const App& a) {
    FileStorage f("workspace_lighting.yml", FileStorage::WRITE);
    if (!f.isOpened()) throw runtime_error("Cannot save lighting selection");
    f << "schema" << 1 << "selected_profile" << a.lightProfile;
}

void bindDynamicReference(App& a) {
    if (!a.referenceDynamic) return;
    auto& v = a.stackVision;
    a.v10.emptyBackground.release();
    a.v10.workspaceChecked = false;
    if (!v.configured || !a.calibration || !a.calibration->valid ||
        norm(Mat(v.captureMapping) - Mat(a.calibration->mapping.h)) > 1e-8)
        return;
    a.v10.emptyBackground = v.reference.clone();
    a.v10.backgroundMapping = v.captureMapping;
    a.v10.backgroundRange = a.v10.enlargedReturns
        ? array<double, 4>{{a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY}}
    : array<double, 4>{ {a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY} };
    a.v10.workspaceChecked = true;
}
void teachStackView(App& a) {
    if (a.busy || a.sentCount || a.v10.phase || a.dynamicStep || a.buildCount || a.dynamicRequest ||
        a.v10.startRequested) {
        a.message = a.v10.status = "Finish/recover current session before teaching stack view";
        return;
    }
    if (a.calibration && (a.calibration->entering || a.calibration->diagnosticRemaining)) {
        a.message = a.v10.status = "Finish/cancel coordinate entry or detection capture first";
        return;
    }
    if (a.lastFrame.empty()) {
        a.message = a.v10.status = "Need live camera first";
        return;
    }
    if (!a.calibration || !a.calibration->valid || a.calibration->freeze) {
        a.message = a.v10.status = "Fit + Save calibration first; use live camera";
        return;
    }
    try {
        Mat frame = a.lastFrame.clone();
        auto h = a.calibration->mapping.h;
        auto projectedBox = [&](double x0, double x1, double y0, double y1) {
            vector<Point2f> pts;
            for (auto w : vector<Point2d>{ {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1} }) {
                auto q = calibrationui::project(h.inv(), w);
                if (!isfinite(q.x) || !isfinite(q.y)) throw runtime_error("Invalid projected workspace");
                pts.emplace_back(float(q.x), float(q.y));
            }
            return boundingRect(pts) & Rect(0, 0, frame.cols, frame.rows);
            };
        // Physical stackSite=1 in the matching MOD; S1/S2/S3 select colour patterns.
        Rect stack = projectedBox(372.65 - 55, 372.65 + 55, 185.98 - 55, 185.98 + 55);
        Rect workspace = projectedBox(min(a.hoverMinX, a.v10.minX), max(a.hoverMaxX, a.v10.maxX),
            min(a.hoverMinY, a.v10.minY), max(a.hoverMaxY, a.v10.maxY)) |
            stack;
        if (stack.width < 60 || stack.height < 60 || workspace.area() < 6400)
            throw runtime_error("Stack/workspace is not sufficiently visible; check camera view");
        auto objects = calibrationui::detect(frame, stackColourModel(a));
        StackVision v;
        v.workspace = workspace;
        v.stack = stack;
        v.reference = frame.clone();
        vector<double> areas;
        for (const auto& d : objects) {
            if ((d.box & workspace).area() > 0)
                throw runtime_error("Remove ALL blocks from pickup and stack areas before Capture Clear Reference");
        }
        if (!areas.empty()) {
            sort(areas.begin(), areas.end());
            v.faceArea = areas[areas.size() / 2];
        }
        else if (a.calibration && a.calibration->valid) {
            auto h = a.calibration->mapping.h;
            auto centre = calibrationui::project(h, Point2d(stack.x + stack.width * .5, stack.y + stack.height * .5));
            vector<Point2f> corners;
            for (auto d : vector<Point2d>{ {-18.5, -18.5}, {18.5, -18.5}, {18.5, 18.5}, {-18.5, 18.5} }) {
                auto q = calibrationui::project(h.inv(), centre + d);
                corners.emplace_back(float(q.x), float(q.y));
            }
            v.faceArea = abs(contourArea(corners));
        }
        if (!isfinite(v.faceArea) || v.faceArea < 50 || v.faceArea >= stack.area() * .5)
            throw runtime_error("Stack ROI needs a larger guard area / visible block size is too small");
        if (!imwrite(visionImage(a), v.reference)) throw runtime_error("Cannot save stack reference image");
        FileStorage f(visionSettings(a), FileStorage::WRITE);
        if (!f.isOpened()) throw runtime_error("Cannot save stack view settings");
        f << "schema" << 3 << "light_profile" << a.lightProfile << "dynamic_profile" << int(a.referenceDynamic)
            << "mapping" << Mat(h) << "hover_bounds" << vector<double>{a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY}
        << "original_bounds" << vector<double>{a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY} << "workspace"
            << workspace << "stack" << stack << "face_area" << v.faceArea << "reference_blocks" << "[";
        for (auto r : v.referenceBlocks) f << r;
        f << "]";
        f.release();
        v.originalBounds = { {a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY} };
        v.savedReference = true;
        v.captureMapping = h;
        v.captureBounds = { {a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY} };
        v.configured = true;
        v.note = "Stack/workspace reference saved; collecting fresh observations";
        a.stackVision = v;
        a.visionProfiles[visionProfileSlot(a)] = v;
        bindDynamicReference(a);
        a.calibration->message = a.message = a.v10.status =
            string(a.referenceDynamic ? "DYNAMIC" : "STATIC") + " " + lightName(a) +
            " reference saved. Reused across cycles/restarts for this lighting.";
    }
    catch (const exception& e) {
        a.stackVision.configured = false;
        a.message = a.v10.status = e.what();
    }
}
void loadStackView(App& a) {
    if (!fileutil::exists(visionSettings(a))) return;
    try {
        FileStorage f(visionSettings(a), FileStorage::READ);
        StackVision v;
        int schema = 0, light = -1, mode = -1;
        f["light_profile"] >> light;
        f["dynamic_profile"] >> mode;
        Mat savedH;
        vector<double> bounds, original;
        f["original_bounds"] >> original;
        f["schema"] >> schema;
        f["mapping"] >> savedH;
        f["hover_bounds"] >> bounds;
        if (original != vector<double>{a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY} || schema != 3 ||
            light != a.lightProfile || mode != int(a.referenceDynamic) || !a.calibration || !a.calibration->valid ||
            savedH.rows != 3 || savedH.cols != 3 || norm(savedH - Mat(a.calibration->mapping.h)) > 1e-8 ||
            bounds != vector<double>{a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY})
            return;
        v.reference = imread(visionImage(a));
        f["workspace"] >> v.workspace;
        f["stack"] >> v.stack;
        f["face_area"] >> v.faceArea;
        for (auto item : f["reference_blocks"]) {
            Rect r;
            item >> r;
            v.referenceBlocks.push_back(r);
        }
        v.configured = !v.reference.empty() && v.reference.size() == a.calibration->mapping.resolution &&
            std::isfinite(v.faceArea) && v.faceArea >= 50 && v.faceArea < v.stack.area() &&
            v.workspace.area() > 0 && v.stack.area() > 0 &&
            (v.workspace & Rect(0, 0, v.reference.cols, v.reference.rows)) == v.workspace &&
            (v.stack & v.workspace) == v.stack;
        v.originalBounds = { {a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY} };
        v.savedReference = true;
        v.captureMapping = a.calibration->mapping.h;
        v.captureBounds = { {a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY} };
        a.stackVision = v;
        a.visionProfiles[visionProfileSlot(a)] = v;
        bindDynamicReference(a);
    }
    catch (const exception& e) {
        a.stackVision.configured = false;
        a.stackVision.note = e.what();
    }
}
void selectVisionProfile(App& a, bool dynamic) {
    if (a.referenceDynamic == dynamic) return;
    a.visionProfiles[visionProfileSlot(a)] = a.stackVision;
    a.referenceDynamic = dynamic;
    a.stackVision = StackVision{};
    loadStackView(a);
    a.stackVision.resetFresh();
    bindDynamicReference(a);
    if (!a.stackVision.configured)
        a.message = a.v10.status =
        string(dynamic ? "DYNAMIC" : "STATIC") +
        " reference missing or mismatched: open Calibration and capture this mode with its normal fixture installed";
}
bool lightingIdle(const App& a) {
    return !a.busy && !a.sentCount && !a.dynamicStep && !a.dynamicRequest && !a.buildCount && !a.v10.phase &&
        !a.v10.startRequested &&
        (!a.calibration || (!a.calibration->entering && !a.calibration->diagnosticRemaining));
}
void lightingMessage(App& a, const string& message) {
    a.message = a.dynamicStatus = a.v10.status = message;
    if (a.calibration) a.calibration->message = message;
}
void cancelLightingCapture(App& a, const string& why) {
    a.lightCaptureActive = false;
    a.lightCaptureCount = 0;
    a.lightCaptureSum.release();
    a.lightCapturePrevious.release();
    a.lightCaptureAnchor.release();
    lightingMessage(a, why);
}
bool selectLightingProfile(App& a, int profile) {
    if (profile < 0 || profile > 1) return false;
    if (!lightingIdle(a) || a.lightCaptureActive) {
        lightingMessage(a, "Finish/recover the current session or cancel capture before changing lighting.");
        return false;
    }
    if (a.lightProfile == profile) return true;
    a.visionProfiles[visionProfileSlot(a)] = a.stackVision;
    a.lightProfile = profile;
    calibrationui::detectionTuning().active = profile;
    a.stackVision = StackVision{};
    a.v10.emptyBackground.release();
    a.v10.workspaceChecked = false;
    loadStackView(a);
    a.stackVision.resetFresh();
    a.stackVision.armed = false;
    bindDynamicReference(a);
    a.v10.planHistory.clear();
    a.v10.planTimes.clear();
    a.v10.stableFrames = 0;
    a.v10.planOutliers = 0;
    for (auto& e : a.v10.returnEvidence) e.clear();
    for (auto& e : a.v10.sourceEvidence) e.clear();
    if (a.calibration) {
        a.calibration->detectionFilter.clear();
        a.calibration->selectedStability.clear();
    }
    try {
        saveLightingPreference(a);
    }
    catch (const exception& e) {
        lightingMessage(a, e.what());
        return false;
    }
    lightingMessage(a, lightName(a) +
        string(a.stackVision.configured
            ? " selected. Matching saved reference loaded; collecting fresh evidence."
            : " selected. Remove all blocks and arm, then capture this empty workspace."));
    return true;
}
// Absolute background comparison: no brightness normalization that could hide a lamp change.
// Tile consensus tolerates small block occlusions; the independent obstruction monitor remains mandatory.
double lightingMatchScore(const Mat& frame, const StackVision& ref) {
    if (!ref.configured || frame.size() != ref.reference.size()) return 1e9;
    Rect roi = ref.workspace & Rect(0, 0, frame.cols, frame.rows);
    if (roi.area() < 6400) return 1e9;
    Mat live, base;
    GaussianBlur(frame(roi), live, { 5, 5 }, 0);
    GaussianBlur(ref.reference(roi), base, { 5, 5 }, 0);
    vector<double> errors;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            Rect tile(x * roi.width / 8, y * roi.height / 8, (x + 1) * roi.width / 8 - x * roi.width / 8,
                (y + 1) * roi.height / 8 - y * roi.height / 8);
            Mat delta;
            absdiff(live(tile), base(tile), delta);
            Scalar d = mean(delta);
            errors.push_back(max(d[0], max(d[1], d[2])));
        }
    sort(errors.begin(), errors.end());
    // At least 80% of spatially distributed tiles must agree with the saved absolute appearance.
    return errors[51];
}
Scalar v13ExposureShift(const Mat& reference, const Mat& frame);
void tickAutomaticLighting(App& a, const Mat& frame, chrono::steady_clock::time_point now) {
    if (!a.calibration || !a.calibration->valid || a.calibration->freeze || a.calibration->opt.source != "camera" ||
        frame.empty())
        return;
    if (a.lightCaptureActive || a.lightingSetupOpen) return;
    // Never replace evidence or the reference underlying an executing transfer.
    if (a.busy || a.sentCount || a.dynamicStep || a.dynamicRequest || a.buildCount || a.v10.phase ||
        a.calibration->entering || a.calibration->diagnosticRemaining) {
        a.autoLightCandidate = -1;
        a.autoLightFrames = 0;
        return;
    }
    a.autoLightChecked = true;
    if (chrono::duration<double>(now - a.autoLightLast).count() < .10) return;
    a.autoLightLast = now;
    a.visionProfiles[visionProfileSlot(a)] = a.stackVision;
    int offset = a.referenceDynamic ? 2 : 0;
    if (chrono::duration<double>(now - a.autoLightLoad).count() > 2) {
        a.autoLightLoad = now;
        for (int p = 0; p < 2; ++p)
            if (!a.visionProfiles[offset + p].configured) {
                App probe;
                probe.calibration = a.calibration;
                probe.referenceDynamic = a.referenceDynamic;
                probe.lightProfile = p;
                probe.hoverMinX = a.hoverMinX;
                probe.hoverMaxX = a.hoverMaxX;
                probe.hoverMinY = a.hoverMinY;
                probe.hoverMaxY = a.hoverMaxY;
                probe.v10.minX = a.v10.minX;
                probe.v10.maxX = a.v10.maxX;
                probe.v10.minY = a.v10.minY;
                probe.v10.maxY = a.v10.maxY;
                loadStackView(probe);
                a.visionProfiles[offset + p] = probe.stackVision;
            }
    }
    // V13: RAW scores decide WHICH reference (a lamp change is never normalised away); the match test
    // uses DRIFT-COMPENSATED scores, so camera auto-exposure / white-balance drift with the lamp off no
    // longer blocks the match. Blocks, the arm or a moved camera change individual tiles and still fail.
    double score[2], matched[2];
    for (int p = 0; p < 2; ++p) {
        auto& r = a.visionProfiles[offset + p];
        bool valid = r.configured &&
            r.captureBounds == array<double, 4>{{a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY}}&&
            r.originalBounds == array<double, 4>{{a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY}}&&
            norm(Mat(r.captureMapping) - Mat(a.calibration->mapping.h)) < 1e-8;
        score[p] = valid ? lightingMatchScore(frame, r) : 1e9;
        matched[p] = 1e9;
        if (valid) {
            Scalar drift = v13ExposureShift(r.reference, frame);
            // Only camera drift (<= 25 levels per channel) is compensated; a bigger change is another lighting state.
            if (max(abs(drift[0]), max(abs(drift[1]), abs(drift[2]))) <= 25) {
                StackVision adjusted = r;
                Mat shifted;
                add(r.reference, drift, shifted);
                adjusted.reference = shifted;
                matched[p] = lightingMatchScore(frame, adjusted);
            }
            else
                matched[p] = score[p];
        }
        a.autoLightRaw[p] = score[p];
        a.autoLightComp[p] = matched[p];
    }
    int best = score[0] < score[1] ? 0 : 1;
    // Similar references are harmless if the currently selected reference itself matches.
    if (matched[a.lightProfile] <= lightMatchLimit() && score[1 - a.lightProfile] - score[a.lightProfile] < 6) best = a.lightProfile;
    if (matched[best] > lightMatchLimit() || (best != a.lightProfile && score[1 - best] - score[best] < 6)) {
        a.autoLightReady = false;
        a.autoLightCandidate = -1;
        a.autoLightFrames = 0;
        return;
    }
    if (a.autoLightCandidate != best) {
        a.autoLightCandidate = best;
        a.autoLightFrames = 0;
        a.autoLightSince = now;
        a.autoLightReady = false;
    }
    ++a.autoLightFrames;
    if (a.autoLightFrames < 5 || chrono::duration<double>(now - a.autoLightSince).count() < .6) return;
    if (best != a.lightProfile) {
        a.v10.startRequested = false;
        if (!selectLightingProfile(a, best)) {
            a.autoLightReady = false;
            return;
        }
        lightingMessage(a,
            string(a.lightProfile == 0 ? "Light on" : "Light off") + " detected. Checking the workspace.");
    }
    a.autoLightReady = true;
}
void beginLightingCapture(App& a, int profile) {
    if (!selectLightingProfile(a, profile)) return;
    if (!lightingIdle(a) || !a.calibration || !a.calibration->valid || a.calibration->freeze ||
        a.calibration->frame.empty() || a.calibration->opt.source != "camera") {
        lightingMessage(a, "Capture needs an idle session, saved camera mapping and live unfrozen camera.");
        return;
    }
    a.lightCaptureActive = true;
    a.lightCaptureCount = 0;
    a.lightCaptureSum.release();
    a.lightCapturePrevious.release();
    a.lightCaptureAnchor.release();
    a.lightCaptureMapping = a.calibration->mapping.h;
    a.lightCaptureSince = chrono::steady_clock::now();
    lightingMessage(
        a, "CAPTURE " + lightName(a) +
        ": remove ALL blocks and arm; keep the light state and camera still. Collecting12 stable images.");
}
Rect lightingWorkspace(const App& a) {
    const auto& c = *a.calibration;
    auto inverse = c.mapping.h.inv();
    vector<Point2f> pixels;
    auto box = [&](double x0, double x1, double y0, double y1) {
        for (auto world : vector<Point2d>{ {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1} }) {
            auto px = calibrationui::project(inverse, world);
            if (!isfinite(px.x) || !isfinite(px.y)) throw runtime_error("Invalid workspace projection");
            pixels.emplace_back(float(px.x), float(px.y));
        }
        };
    box(min(a.hoverMinX, a.v10.minX), max(a.hoverMaxX, a.v10.maxX), min(a.hoverMinY, a.v10.minY),
        max(a.hoverMaxY, a.v10.maxY));
    box(372.65 - 55, 372.65 + 55, 185.98 - 55, 185.98 + 55);
    Rect workspace = boundingRect(pixels) & Rect(0, 0, c.frame.cols, c.frame.rows);
    if (workspace.area() < 6400) throw runtime_error("Workspace not sufficiently visible for lighting capture");
    return workspace;
}
void tickLightingCapture(App& a, chrono::steady_clock::time_point now = chrono::steady_clock::now()) {
    if (!a.lightCaptureActive) return;
    if (!lightingIdle(a) || !a.calibration || !a.calibration->valid || a.calibration->freeze ||
        a.calibration->frame.empty() || norm(Mat(a.lightCaptureMapping) - Mat(a.calibration->mapping.h)) > 1e-8) {
        cancelLightingCapture(a, "Lighting capture cancelled: session/camera/calibration changed.");
        return;
    }
    auto& c = *a.calibration;
    const Mat& frame = c.frame;
    if (frame.size() != c.mapping.resolution) {
        cancelLightingCapture(a, "Capture cancelled: resolution differs from saved mapping.");
        return;
    }
    Rect workspace = lightingWorkspace(a);
    auto objects = calibrationui::detect(frame, stackColourModel(a));
    // Do not absorb any detected painted object into either empty reference.
    if (any_of(objects.begin(), objects.end(),
        [&](const calibrationui::Detection& d) { return (d.box & workspace).area() > 0; })) {
        cancelLightingCapture(a, "Capture refused: remove ALL coloured blocks/arm from the workspace first.");
        return;
    }
    if (!a.lightCapturePrevious.empty()) {
        if (a.lightCapturePrevious.size() != frame.size()) {
            cancelLightingCapture(a, "Capture cancelled: camera resolution changed.");
            return;
        }
        // Compare to the first stable image, not just the previous image: slow exposure drift must settle too.
        // Blur only for motion detection so normal low-light sensor noise does not reset capture forever.
        Mat before, after, delta, gray, mask;
        GaussianBlur(a.lightCaptureAnchor(workspace), before, { 5, 5 }, 0);
        GaussianBlur(frame(workspace), after, { 5, 5 }, 0);
        absdiff(before, after, delta);
        cvtColor(delta, gray, COLOR_BGR2GRAY);
        threshold(gray, mask, 12, 255, THRESH_BINARY);
        if (double(countNonZero(mask)) / workspace.area() > .01) {
            a.lightCaptureCount = 0;
            a.lightCaptureSum.release();
            a.lightCaptureSince = now;
            a.lightCapturePrevious = frame.clone();
            a.lightCaptureAnchor = frame.clone();
            lightingMessage(a, "Lighting/image is still settling or moving. Waiting for12 stable empty images.");
            return;
        }
    }
    a.lightCapturePrevious = frame.clone();
    if (a.lightCaptureAnchor.empty()) a.lightCaptureAnchor = frame.clone();
    if (a.lightCaptureSum.empty()) a.lightCaptureSum = Mat::zeros(frame.size(), CV_32FC3);
    accumulate(frame, a.lightCaptureSum);
    ++a.lightCaptureCount;
    lightingMessage(a, "CAPTURE " + lightName(a) + ": " + to_string(min(12, a.lightCaptureCount)) +
        "/12 stable images; keep workspace empty.");
    if (a.lightCaptureCount < 12 || chrono::duration<double>(now - a.lightCaptureSince).count() < .5) return;
    Mat averaged;
    a.lightCaptureSum.convertTo(averaged, CV_8UC3, 1.0 / a.lightCaptureCount);
    a.lightCaptureActive = false;
    a.lightCaptureCount = 0;
    a.lightCaptureSum.release();
    a.lightCapturePrevious.release();
    a.lightCaptureAnchor.release();
    Mat live = a.lastFrame;
    a.lastFrame = averaged;
    teachStackView(a);
    a.lastFrame = live;
    if (a.stackVision.configured)
        lightingMessage(
            a, lightName(a) +
            " empty reference saved from stable images. Automatic matching resumes when this setup is closed.");
}
void openLightingSetup(App& a) {
    if (!lightingIdle(a) || a.lightCaptureActive) {
        lightingMessage(a, "Lighting setup is available only between cycles; finish/recover or cancel capture first.");
        return;
    }
    a.lightingSetupOpen = true;
    lightingMessage(
        a,
        "Capture empty workspace separately with LIGHT ON and LIGHT OFF. Keep camera fixed; remove blocks and arm. Selection does not switch the physical lamp.");
}
void drawLightingSetup(Mat& ui, const App& a) {
    if (a.tuningOpen) {
        drawTuningPage(ui, const_cast<App&>(a));
        return;
    }
    panel(ui, { 850, 100, 415, 615 });
    label(ui, "WORKSPACE LIGHTING", { 865, 132 }, .55, WHITE, 2);
    button(ui, { 865, 155, 185, 36 }, "STATIC", !a.referenceDynamic);
    button(ui, { 1060, 155, 185, 36 }, "DYNAMIC", a.referenceDynamic);
    button(ui, { 865, 207, 185, 38 }, "LIGHT ON", a.lightProfile == 0);
    button(ui, { 1060, 207, 185, 38 }, "LIGHT OFF", a.lightProfile == 1);
    label(ui, "Remove ALL blocks and arm before either capture.", { 865, 274 }, .35, GOLD);
    label(ui, "Set the actual lamp first; these buttons do not control it.", { 865, 297 }, .32, MUTED);
    button(ui, { 865, 325, 380, 44 }, "CAPTURE LIGHT ON");
    button(ui, { 865, 385, 380, 44 }, "CAPTURE LIGHT OFF");
    label(ui, "ACTIVE: " + lightName(a), { 865, 464 }, .45, GOLD);
    label(ui,
        a.stackVision.configured ? "Saved reference loaded for this mode and lighting."
        : "No matching reference yet: capture this lighting.",
        { 865, 490 }, .34, MUTED);
    if (a.lightCaptureActive)
        label(ui, "CAPTURING " + to_string(min(12, a.lightCaptureCount)) + "/12 stable frames", { 865, 521 }, .4, GOLD);
    button(ui, { 865, 550, 380, 40 }, a.lightCaptureActive ? "CANCEL CAPTURE" : "CLOSE LIGHTING SETUP");
    button(ui, { 865, 600, 380, 36 }, "DETECTION + OFFSET TUNING", !a.lightCaptureActive);
    label(ui, "One shared XY calibration; four independent references:", { 865, 660 }, .33, MUTED);
    label(ui, "Auto selects a stable match between cycles.", { 865, 683 }, .32, MUTED);
}
bool lightingMouse(App& a, int x, int y) {
    if (!a.lightingSetupOpen) return false;
    Point click(x, y);
    if (a.tuningOpen) {
        if (y < 100) {
            a.lightingSetupOpen = false;
            a.tuningOpen = false;
            return false;
        }
        return tuningMouse(a, x, y);
    }
    if (Rect(865, 550, 380, 40).contains(click)) {
        if (a.lightCaptureActive)
            cancelLightingCapture(a, "Lighting capture cancelled. Saved references retained.");
        else
            a.lightingSetupOpen = false;
        return true;
    }
    if (a.lightCaptureActive) {
        lightingMessage(a, "Capture in progress: keep the workspace empty or press CANCEL CAPTURE.");
        return true;
    }
    if (y < 100) {
        a.lightingSetupOpen = false;
        a.tuningOpen = false;
        return false;
    }
    if (Rect(865, 600, 380, 36).contains(click)) {
        a.tuningOpen = true;
        lightingMessage(a, "Tuning: S/V rows apply to " + lightName(a) + ". Switch lighting with BACK if needed.");
        return true;
    }
    if (Rect(865, 155, 185, 36).contains(click) || Rect(1060, 155, 185, 36).contains(click)) {
        if (!lightingIdle(a)) {
            lightingMessage(a, "Finish/recover the active session before changing reference mode.");
            return true;
        }
        selectVisionProfile(a, x >= 1060);
        lightingMessage(a, string(a.referenceDynamic ? "DYNAMIC " : "STATIC ") + lightName(a) +
            " empty-reference setup selected.");
        return true;
    }
    if (Rect(865, 207, 185, 38).contains(click)) {
        selectLightingProfile(a, 0);
        return true;
    }
    if (Rect(1060, 207, 185, 38).contains(click)) {
        selectLightingProfile(a, 1);
        return true;
    }
    if (Rect(865, 325, 380, 44).contains(click)) {
        beginLightingCapture(a, 0);
        return true;
    }
    if (Rect(865, 385, 380, 44).contains(click)) {
        beginLightingCapture(a, 1);
        return true;
    }
    return true;
}
void drawStackVision(Mat& frame, const App& a) {
    const auto& v = a.stackVision;
    if (!v.configured || frame.size() != v.reference.size()) return;
    if (!v.clearStable && !v.obstacleMask.empty()) {
        vector<vector<Point>> contours;
        findContours(v.obstacleMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        sort(contours.begin(), contours.end(),
            [](const vector<Point>& x, const vector<Point>& y) { return contourArea(x) > contourArea(y); });
        int count = 0;
        for (const auto& q : contours)
            if (contourArea(q) >= 20 && count++ < 3) {
                Rect r = boundingRect(q);
                rectangle(frame, r, Scalar(0, 165, 255), 2);
            }
    }
    rectangle(frame, v.workspace, v.clearStable ? Scalar(140, 200, 70) : Scalar(40, 100, 235), 1);
    rectangle(frame, v.stack, Scalar(230, 210, 30), 2);
    label(frame,
        (v.finalTopOnly ? string("FINAL TOP / ") : "STACK " + to_string(v.progress) + "/3 ") +
        (v.clearStable ? "WORKSPACE CLEAR" : "WORKSPACE UNKNOWN/BUSY"),
        { v.stack.x, max(15, v.stack.y - 8) }, .38, Scalar(230, 210, 30));
}

// ------------------------------------------------------------
// SECTION 6  STATIC MODE OPERATOR ACTIONS
//    Start, return and release for the nine taught positions P1 to P9.
// ------------------------------------------------------------
bool staticSourcesGone(const App& a, int routine) {
    if (routine < 0 || routine >= 3) return false;
    for (int src : a.sentSources[routine])
        if (src < 0 || src >= N || a.positions[src].uncertain || a.positions[src].confirmed != UNKNOWN) return false;
    return true;
}
bool staticSourcesCurrent(const App& a, const Plan& p) {
    if (!p.valid) return false;
    for (int l = 0; l < 3; ++l) {
        int src = p.source[l];
        if (src < 0 || src >= N || a.positions[src].uncertain || a.positions[src].confirmed != p.colours[l])
            return false;
    }
    return true;
}
bool staticReturnClear(const App& a, int s) {
    for (int d : a.returnDest[s])
        if (d < 0 || d >= N || a.positions[d].uncertain || a.positions[d].confirmed != UNKNOWN) return false;
    return true;
}

// STACK PLAN frame (240 ... 241): A1..A3 stack, B1..B9 level 1 pickup, C1..C9 level 2, D1..D9 level 3.
void operatorStart(App& a) {
    if (a.lightCaptureActive) {
        a.message = "Finish/cancel lighting capture before starting motion";
        return;
    }
    if (a.v10.phase) {
        a.message = "Unresolved V10 dynamic cycle: inspect/recover before static operation";
        return;
    }
    if (a.calibrationMode) return;
    if (a.busy) return;
    a.releaseArmed = false;
    if (a.learnPick || a.learning || a.mapping || a.sampleMode || a.zoneSetup) {
        a.message = "Finish setup before starting";
        return;
    }
    if (!a.serialConnected) {
        a.message = "START BLOCKED: COM OFFLINE";
        return;
    }
    if (a.commandLatched) {
        a.message = "START BLOCKED: press C to clear first; robot status UNKNOWN";
        return;
    }
    if (!a.planRefreshReady) {
        a.message = "START BLOCKED: wait for vision refresh or press P";
        return;
    }
    if (a.sentCount > 0) {
        a.message = "WAIT: automatic stack/return cycle in progress";
        return;
    }
    if (a.sentPlans[a.selected - 1]) {
        a.message = "S" + to_string(a.selected) + " ALREADY SENT; send return (D) then release (T)";
        return;
    }
    if (!a.verifiedMap || a.mapped != N) {
        a.message = "START BLOCKED: verify P1-P9 map";
        return;
    }
    auto visualStart = stackStartReason(a);
    if (!visualStart.empty()) {
        a.message = visualStart;
        return;
    }
    Plan p = a.queued[a.selected - 1];
    if (!p.valid) {
        a.message = "START BLOCKED: " + p.message;
        return;
    }
    if (!staticSourcesCurrent(a, p)) {
        a.message = "START BLOCKED: selected blocks changed; wait for fresh vision plan";
        return;
    }
    for (int level = 0; level < 3; ++level) {
        const int src = p.source[level];
        if (src < 0 || src >= N || a.reservedSources[src] || (!a.allowRepeatPickups && a.processedBlocks[src])) {
            a.message = "START BLOCKED: source already used or invalid; refresh plan";
            return;
        }
        for (int earlier = 0; earlier < level; ++earlier)
            if (p.source[earlier] == src) {
                a.message = "START BLOCKED: duplicate pickup in plan";
                return;
            }
    }
    const vector<uint8_t> fields = {
        static_cast<uint8_t>(0xA0 + a.selected), static_cast<uint8_t>(0xB0 + p.source[0] + 1),
        static_cast<uint8_t>(0xC0 + p.source[1] + 1), static_cast<uint8_t>(0xD0 + p.source[2] + 1) };
    array<int, 3> expected;
    for (int i = 0; i < 3; ++i) expected[i] = p.colours[i] - 1;
    a.stackVision.begin(expected);
    if (!sendFrame(a, 240, fields, 241)) return;
    a.commandLatched = true;
    for (int source : p.source) a.reservedSources[source] = true;
    refreshPlans(a);
    a.sentPlans[a.selected - 1] = true;
    a.sentSources[a.selected - 1] = p.source;
    a.sentColours[a.selected - 1] = p.colours;
    a.emptySince[a.selected - 1] = {};
    a.restoredSince[a.selected - 1] = {};
    a.returnStarted[a.selected - 1] = false;
    a.stackSentAt[a.selected - 1] = chrono::steady_clock::now();
    a.sendOrder[a.sentCount++] = a.selected;
    a.message = "STACK PLAN SENT (NO ACK): S" + to_string(a.selected) + " P" + to_string(p.source[0] + 1) + " P" +
        to_string(p.source[1] + 1) + " P" + to_string(p.source[2] + 1) + "; RX expected: 240," +
        to_string(160 + a.selected) + "," + to_string(177 + p.source[0]) + "," + to_string(193 + p.source[1]) +
        "," + to_string(209 + p.source[2]) + ",241";
}

// RANDOM RETURN frame (250 ... 251), sent AFTER the pendant shows "1=WAIT FOR RETURN PLAN":
//   0x80+S = stack, 0x10+P = level-3 drop, 0x20+P = level-2 drop, 0x30+P = level-1 drop.
// Destinations are chosen at random among positions that vision currently confirms EMPTY
// and that no other pending return has claimed. The robot unstacks top-first (L3, L2, L1).
void operatorReturn(App& a) {
    if (a.calibrationMode) return;
    if (a.busy) return;
    a.releaseArmed = false;
    if (!a.serialConnected) {
        a.message = "RETURN BLOCKED: COM OFFLINE";
        return;
    }
    const int s = a.selected - 1;
    if (!a.sentPlans[s]) {
        a.message = "S" + to_string(a.selected) + " NOT SENT - nothing to return";
        return;
    }
    if (!staticSourcesGone(a, s)) {
        a.message = "RETURN BLOCKED: selected pickup blocks must be confirmed absent";
        return;
    }
    if (!a.stackVision.settledReturnReady(chrono::steady_clock::now())) {
        a.message = "RETURN BLOCKED: " + a.stackVision.note;
        return;
    }
    if (a.returnSent[s]) {
        a.message = "S" + to_string(a.selected) + " RETURN ALREADY SENT; after robot finishes press T";
        return;
    }
    if (!a.verifiedMap || a.mapped != N) {
        a.message = "RETURN BLOCKED: verify P1-P9 map";
        return;
    }
    vector<int> freePos;
    for (int i = 0; i < N; ++i)
        if (!a.positions[i].uncertain && a.positions[i].confirmed == UNKNOWN && !a.returnReserved[i])
            freePos.push_back(i);
    if (freePos.size() < 3) {
        a.message = "RETURN BLOCKED: need 3 stable EMPTY positions (have " + to_string(freePos.size()) + ")";
        return;
    }
    shuffle(freePos.begin(), freePos.end(), a.rng);
    const array<int, 3> dest{ {freePos[0], freePos[1], freePos[2]} }; // level 1,2,3 drop positions
    if (a.commandLatched) {
        operatorClear(a);
        if (a.commandLatched) return;
    }
    const vector<uint8_t> fields = { static_cast<uint8_t>(0x80 + a.selected), static_cast<uint8_t>(0x10 + dest[2] + 1),
                                    static_cast<uint8_t>(0x20 + dest[1] + 1), static_cast<uint8_t>(0x30 + dest[0] + 1) };
    if (!sendFrame(a, 250, fields, 251)) return;
    a.commandLatched = true;
    a.returnDest[s] = dest;
    a.returnSent[s] = true;
    a.retryAt[s] = chrono::steady_clock::now();
    for (int d : dest) a.returnReserved[d] = true;
    a.message = "RETURN PLAN SENT (NO ACK): S" + to_string(a.selected) + " drops L3>P" + to_string(dest[2] + 1) +
        " L2>P" + to_string(dest[1] + 1) + " L1>P" + to_string(dest[0] + 1) + "; VERIFY ON PENDANT";
}

// Call ONLY after the robot has physically finished the return.
// Frees the stack, its pickup reservations and its drop reservations, then vision re-reads P1-P9.
void operatorRelease(App& a) {
    if (a.calibrationMode) return;
    if (a.busy) return;
    const int s = a.selected - 1;
    if (!a.sentPlans[s]) {
        a.message = "S" + to_string(a.selected) + " NOT SENT - nothing to release";
        return;
    }
    if (!a.returnSent[s] && !a.releaseArmed) {
        a.releaseArmed = true;
        a.message = "NO RETURN PLAN SENT FOR S" + to_string(a.selected) + ": press T again to FORCE release";
        return;
    }
    // Mark the returned blocks at their NEW locations, not the old pickup IDs.
    // With fewer than nine blocks, an old source can later hold an unused block.
    for (int src : a.sentSources[s])
        if (src >= 0 && src < N) {
            a.reservedSources[src] = false;
            a.processedBlocks[src] = false;
        }
    for (int dest : a.returnDest[s])
        if (dest >= 0 && dest < N) a.processedBlocks[dest] = true;
    for (int d : a.returnDest[s])
        if (d >= 0 && d < N) a.returnReserved[d] = false;
    a.returnDest[s] = { {-1, -1, -1} };
    a.returnSent[s] = false;
    a.sentPlans[s] = false;
    ++a.completedCycles;
    a.sentSources[s] = { {-1, -1, -1} };
    array<int, 3> o{ {0, 0, 0} };
    int n = 0;
    for (int j = 0; j < a.sentCount; ++j)
        if (a.sendOrder[j] != a.selected) o[n++] = a.sendOrder[j];
    a.sendOrder = o;
    a.sentCount = n;
    a.observedHistory[s].clear();
    a.stackTop[s] = 0;
    a.releaseArmed = false;
    resetObservations(a);       // positions become UNCERTAIN until re-confirmed from the camera
    a.planRefreshReady = false; // fresh plan only after vision has re-read the new layout
    refreshPlans(a);
    bool saved = saveConfig(a);
    a.message = saved ? (a.allowRepeatPickups
        ? "Cycle complete. Repeat pickups allowed; USED history retained."
        : "Cycle complete. Returned blocks marked USED; next plan picks unused blocks only.")
        : "Cycle complete; USED history is in memory only: cannot save handeye_config.yml";
}

void newBatch(App& a) {
    if (a.busy || a.sentCount) {
        a.message = "NEW BATCH unavailable during a cycle";
        return;
    }
    a.processedBlocks.fill(false);
    resetObservations(a);
    a.planRefreshReady = false;
    a.message = saveConfig(a) ? "NEW BATCH: all blocks eligible again; waiting for stable camera readings"
        : "New batch reset in memory; settings save failed";
}

void setPickupMode(App& a, bool allowRepeat) {
    if (a.busy || a.sentCount || a.v10.phase || a.dynamicStep || a.buildCount) {
        a.message = "Change pickup mode after the current cycle finishes";
        return;
    }
    a.allowRepeatPickups = allowRepeat;
    refreshPlans(a);
    bool saved = saveConfig(a);
    a.message = allowRepeat ? "ALLOW REPEATS: handled blocks may be picked again"
        : "UNUSED ONLY: handled blocks are excluded; history retained";
    if (!saved) a.message += " (settings save failed)";
}

string startBlock(const App& a) {
    if (a.v10.phase) return "Unresolved dynamic cycle - open Dynamic for recovery status";
    if (a.calibrationMode) return "Calibration preview mode - robot commands disabled";
    if (a.busy) return "Sending plan - please wait";
    if (a.sentCount) return "Automatic cycle in progress";
    if (!a.serialConnected) return "Connect COM: launch with COM port argument";
    if (a.learnPick || a.learning) return "Finish or cancel block measurement";
    if (a.mapping || a.sampleMode || a.zoneSetup) return "Finish setup before starting";
    if (a.mapped != N || !a.verifiedMap) return "Open SETUP and map / verify P1-P9";
    if (a.commandLatched) return "Previous command still latched - check controller";
    if (!a.planRefreshReady) return "Waiting for fresh camera observations";
    auto visual = stackStartReason(a);
    if (!visual.empty()) return visual;
    if (!a.queued[a.selected - 1].valid) return a.queued[a.selected - 1].message;
    return "";
}

// Dynamic single-block test: staged operator actions, never automatic retries.
bool v10Point(const App&, Point2d);
bool v10HoverPoint(const App&, Point2d);
string v10HoverBlockReason(const App&, Point2d);
// Frames 232..233 carry XY at 0.5 mm: X=100+q/2, Y=-400+q/2 (old 230..231 frames used a smaller 0.1 mm range).

// ------------------------------------------------------------
// SECTION 7  PICK OFFSETS AND PICK FACE CHECKS
//    Camera XY plus offsets gives the robot target. Outside the P1 to P9 area the mapping extrapolates,
//    so a small allowance is added there, and blue gets its own Y correction (tuned on the TEST page).
//    The pick face checks refuse blocks that are rotated or whose top face is unclear.
// ------------------------------------------------------------
vector<uint8_t> dynamicFields(int token, int action, Point2d xy) {
    if (token < 1 || token > 127 || action < 1 || action > 8 || !isfinite(xy.x) || !isfinite(xy.y) ||
        !calibrationui::reachOk(xy))
        throw runtime_error("Dynamic XY/action outside the robot's reachable test area");
    int qx = cvRound((xy.x - 100) * 2), qy = cvRound((xy.y + 400) * 2);
    if (qx < 0 || qx > 2047 || qy < 0 || qy > 2047) throw runtime_error("XY cannot be encoded");
    vector<uint8_t> out = { uint8_t(token),        uint8_t(action),       uint8_t(qx / 128 + 1),
                           uint8_t(qx % 128 + 1), uint8_t(qy / 128 + 1), uint8_t(qy % 128 + 1) };
    int sum = 0;
    for (auto b : out) sum += b;
    out.push_back(uint8_t(sum % 251 + 1)); // Keep zero free for the separators between wire values.
    return out;
}
// V13 offset allowance. Inside the P1-P9 Fit + Save box the mapping is fitted on those points; outside
// it extrapolates and the error grows with distance. Positions beyond the box are pushed further away
// from it by gain x distance (mm per mm), per side, capped. 19 Oct jog checks: -Y side (camera view
// LEFT, away from the robot arm) needed about -0.03 mm per mm. Tune in v13_offset_allowance.yml.
struct V13Allowance {
    // V13.9: blueY is now the centre-line Y minus for ALL colours (left of the centre line);
    // centreYPlus is the matching +Y for all colours right of it (0 until tuned with a hover check).
    double xMinus = 0, xPlus = 0, yMinus = 0.03, yPlus = 0, cap = 8, blueY = 0.03, centreYPlus = 0;
    // Per-colour centre-line gains (RED, GREEN, BLUE). NaN = use the shared blueY / centreYPlus above.
    // Set from the TEST page: CL/CR change the gain of the colour of the selected block only.
    // 23 Oct (new camera angle): red is right at the shared 0.03; green and blue were over, so they start lower.
    array<double, 3> colourYMinus{ {NAN, 0.015, 0.010} }, colourYPlus{ {NAN, NAN, NAN} };
    double yMinusFor(int colour) const {
        return colour >= 0 && colour < 3 && isfinite(colourYMinus[colour]) ? colourYMinus[colour] : blueY;
    }
    double yPlusFor(int colour) const {
        return colour >= 0 && colour < 3 && isfinite(colourYPlus[colour]) ? colourYPlus[colour] : centreYPlus;
    }
    string checkedFor;
};
static const char* const V13_COLOUR_KEYS[3] = { "red", "green", "blue" };
V13Allowance& v13Allowance() {
    static V13Allowance g;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        if (std::ifstream("v13_offset_allowance.yml").good()) {
            FileStorage f("v13_offset_allowance.yml", FileStorage::READ);
            if (f.isOpened()) {
                if (!f["gain_x_minus"].empty()) f["gain_x_minus"] >> g.xMinus;
                if (!f["gain_x_plus"].empty()) f["gain_x_plus"] >> g.xPlus;
                if (!f["gain_y_minus"].empty()) f["gain_y_minus"] >> g.yMinus;
                if (!f["gain_y_plus"].empty()) f["gain_y_plus"] >> g.yPlus;
                if (!f["cap_mm"].empty()) f["cap_mm"] >> g.cap;
                if (!f["blue_y_minus_per_mm"].empty()) f["blue_y_minus_per_mm"] >> g.blueY;
                if (!f["centre_y_minus_per_mm"].empty()) f["centre_y_minus_per_mm"] >> g.blueY;
                if (!f["centre_y_plus_per_mm"].empty()) f["centre_y_plus_per_mm"] >> g.centreYPlus;
                for (int k = 0; k < 3; ++k) {
                    string m = string("centre_y_minus_") + V13_COLOUR_KEYS[k], pl = string("centre_y_plus_") + V13_COLOUR_KEYS[k];
                    if (!f[m].empty()) f[m] >> g.colourYMinus[k];
                    if (!f[pl].empty()) f[pl] >> g.colourYPlus[k];
                }
            }
        }
        else {
            FileStorage w("v13_offset_allowance.yml", FileStorage::WRITE);
            if (w.isOpened())
                w << "note"
                << "mm of extra offset per mm OUTSIDE the P1-P9 box, per side; Y minus = camera-view left (away from the arm)"
                << "gain_x_minus" << g.xMinus << "gain_x_plus" << g.xPlus << "gain_y_minus" << g.yMinus
                << "gain_y_plus" << g.yPlus << "cap_mm" << g.cap << "centre_y_minus_per_mm" << g.blueY
                << "centre_y_plus_per_mm" << g.centreYPlus;
        }
        for (double* x : { &g.xMinus, &g.xPlus, &g.yMinus, &g.yPlus, &g.blueY, &g.centreYPlus })
            if (!isfinite(*x) || *x < 0 || *x > .2) *x = 0;
        for (int k = 0; k < 3; ++k) {
            if (isfinite(g.colourYMinus[k]) && (g.colourYMinus[k] < 0 || g.colourYMinus[k] > .2)) g.colourYMinus[k] = NAN;
            if (isfinite(g.colourYPlus[k]) && (g.colourYPlus[k] < 0 || g.colourYPlus[k] > .2)) g.colourYPlus[k] = NAN;
        }
        if (!isfinite(g.cap) || g.cap < 0 || g.cap > 15) g.cap = 8;
        cout << "Offset allowance (mm per mm outside P1-P9): X- " << g.xMinus << " X+ " << g.xPlus << " Y- " << g.yMinus
            << " Y+ " << g.yPlus << ", cap " << g.cap << " mm; centre line Y- R " << g.yMinusFor(0) << " G "
            << g.yMinusFor(1) << " B " << g.yMinusFor(2) << ", Y+ R " << g.yPlusFor(0) << " G " << g.yPlusFor(1) << " B "
            << g.yPlusFor(2) << endl;
    }
    return g;
}
void v13SaveAllowance() {
    const auto& g = v13Allowance();
    FileStorage w("v13_offset_allowance.yml", FileStorage::WRITE);
    if (w.isOpened())
        w << "note"
        << "mm of extra offset per mm OUTSIDE the P1-P9 box, per side; Y minus = camera-view left (away from the arm). Set from the TEST page."
        << "gain_x_minus" << g.xMinus << "gain_x_plus" << g.xPlus << "gain_y_minus" << g.yMinus << "gain_y_plus"
        << g.yPlus << "cap_mm" << g.cap << "centre_y_minus_per_mm" << g.blueY << "centre_y_plus_per_mm" << g.centreYPlus;
    if (w.isOpened())
        for (int k = 0; k < 3; ++k) {
            if (isfinite(g.colourYMinus[k])) w << (string("centre_y_minus_") + V13_COLOUR_KEYS[k]) << g.colourYMinus[k];
            if (isfinite(g.colourYPlus[k])) w << (string("centre_y_plus_") + V13_COLOUR_KEYS[k]) << g.colourYPlus[k];
        }
}
// ---- TUNING page (Calibration setup > Lighting > DETECTION + OFFSET TUNING) --------------------------
// Detection S/V minimums per lighting, the lighting match limit, and the per-colour centre-line offsets.
void saveDetectionTuning() {
    const auto& t = calibrationui::detectionTuning();
    FileStorage f("detection_tuning.yml", FileStorage::WRITE);
    if (!f.isOpened()) return;
    f << "note" << "Set from the TUNING page. -1 = use the calibration value."
        << "match_limit" << t.matchLimit << "sat_min_light_on" << t.satMin[0] << "sat_min_light_off" << t.satMin[1]
        << "val_min_light_on" << t.valMin[0] << "val_min_light_off" << t.valMin[1];
}
void loadDetectionTuning() {
    auto& t = calibrationui::detectionTuning();
    if (!fileutil::exists("detection_tuning.yml")) return;
    try {
        FileStorage f("detection_tuning.yml", FileStorage::READ);
        if (!f["match_limit"].empty()) f["match_limit"] >> t.matchLimit;
        if (!f["sat_min_light_on"].empty()) f["sat_min_light_on"] >> t.satMin[0];
        if (!f["sat_min_light_off"].empty()) f["sat_min_light_off"] >> t.satMin[1];
        if (!f["val_min_light_on"].empty()) f["val_min_light_on"] >> t.valMin[0];
        if (!f["val_min_light_off"].empty()) f["val_min_light_off"] >> t.valMin[1];
    }
    catch (...) {}
    if (!isfinite(t.matchLimit) || t.matchLimit < TUNE_MATCH_MIN || t.matchLimit > TUNE_MATCH_MAX) t.matchLimit = 15;
    for (int k = 0; k < 2; ++k) {
        if (t.satMin[k] != -1 && (t.satMin[k] < TUNE_SAT_MIN || t.satMin[k] > TUNE_SAT_MAX)) t.satMin[k] = -1;
        if (t.valMin[k] != -1 && (t.valMin[k] < TUNE_VAL_MIN || t.valMin[k] > TUNE_VAL_MAX)) t.valMin[k] = -1;
    }
}
// Calibration values that -1 stands for (shown so the operator sees the number in use).
int tuningBaseSat(const App& a) { return a.calibration ? a.calibration->cfg.sat : 55; }
int tuningBaseVal(const App& a) { return a.calibration ? a.calibration->cfg.val : 35; }
struct TuningRow {
    string name, value;
    std::function<void(int)> step; // -1 or +1
};
vector<TuningRow> tuningRows(App& a) {
    auto& t = calibrationui::detectionTuning();
    auto& g = v13Allowance();
    int light = a.lightProfile == 1 ? 1 : 0;
    string tag = light ? " (OFF)" : " (ON)";
    vector<TuningRow> rows;
    rows.push_back({ "Light match limit", calibrationui::fmt(t.matchLimit, 0), [&t](int d) {
        t.matchLimit = max(TUNE_MATCH_MIN, min(TUNE_MATCH_MAX, t.matchLimit + d));
        saveDetectionTuning();
    } });
    rows.push_back({ "Min saturation" + tag,
        t.satMin[light] < 0 ? "calib " + to_string(tuningBaseSat(a)) : to_string(t.satMin[light]),
        [&t, &a, light](int d) {
            int v = t.satMin[light] < 0 ? tuningBaseSat(a) : t.satMin[light];
            t.satMin[light] = max(TUNE_SAT_MIN, min(TUNE_SAT_MAX, v + 5 * d));
            saveDetectionTuning();
            if (a.calibration) a.calibration->detectionFilter.clear();
        } });
    rows.push_back({ "Min brightness V" + tag,
        t.valMin[light] < 0 ? "calib " + to_string(tuningBaseVal(a)) : to_string(t.valMin[light]),
        [&t, &a, light](int d) {
            int v = t.valMin[light] < 0 ? tuningBaseVal(a) : t.valMin[light];
            t.valMin[light] = max(TUNE_VAL_MIN, min(TUNE_VAL_MAX, v + 2 * d));
            saveDetectionTuning();
            if (a.calibration) a.calibration->detectionFilter.clear();
        } });
    rows.push_back({ "Left allowance AL (all)", calibrationui::fmt(g.yMinus, 3), [&g](int d) {
        // Extra Y- per mm OUTSIDE the P1-P9 box on the left, all colours, capped at cap_mm.
        g.yMinus = max(0.0, min(0.2, g.yMinus + .005 * d));
        v13SaveAllowance();
    } });
    const char* names[3] = { "RED", "GREEN", "BLUE" };
    for (int side = 0; side < 2; ++side)
        for (int k = 0; k < 3; ++k) {
            double value = side == 0 ? g.yMinusFor(k) : g.yPlusFor(k);
            rows.push_back({ string(names[k]) + (side == 0 ? " offset Y- (left)" : " offset Y+ (right)"),
                calibrationui::fmt(value, 3), [&g, side, k](int d) {
                    auto& slot = side == 0 ? g.colourYMinus[k] : g.colourYPlus[k];
                    double v = side == 0 ? g.yMinusFor(k) : g.yPlusFor(k);
                    slot = max(0.0, min(0.2, v + .005 * d));
                    v13SaveAllowance();
                } });
        }
    return rows;
}
Rect tuningMinus(int i) { return Rect(1085, 165 + i * 38, 50, 30); }
Rect tuningPlus(int i) { return Rect(1195, 165 + i * 38, 50, 30); }
const Rect TUNING_BACK(865, 640, 380, 40);
void drawTuningPage(Mat& ui, App& a) {
    panel(ui, { 850, 100, 415, 615 });
    label(ui, "DETECTION + OFFSET TUNING", { 865, 132 }, .52, WHITE, 2);
    label(ui, "Active lighting: " + lightName(a) + "  (S/V rows apply to it only)", { 865, 155 }, .33, GOLD);
    auto rows = tuningRows(a);
    for (size_t i = 0; i < rows.size(); ++i) {
        int y = 165 + int(i) * 38;
        if (i == 3 || i == 4) line(ui, { 865, y - 4 }, { 1245, y - 4 }, MUTED, 1);
        label(ui, rows[i].name, { 865, y + 21 }, .38, WHITE);
        button(ui, tuningMinus(int(i)), "-");
        label(ui, rows[i].value, { 1140, y + 21 }, .38, GOLD);
        button(ui, tuningPlus(int(i)), "+");
    }
    label(ui, "Match limit: higher accepts a looser ON/OFF reference match.", { 865, 562 }, .31, MUTED);
    label(ui, "S/V: lower finds darker blocks but risks shadows/tint.", { 865, 580 }, .31, MUTED);
    label(ui, "AL: mm per mm past the P1-P9 box (left). Offsets: from centre line.", { 865, 598 }, .31, MUTED);
    label(ui, "0.005 per press, max 8 mm each. Saved at once; HOVER test after.", { 865, 616 }, .31, MUTED);
    button(ui, TUNING_BACK, "BACK TO LIGHTING");
}
bool tuningMouse(App& a, int x, int y) {
    Point click(x, y);
    if (TUNING_BACK.contains(click)) {
        a.tuningOpen = false;
        return true;
    }
    if (!lightingIdle(a)) {
        lightingMessage(a, "Tuning is locked while a cycle is running.");
        return true;
    }
    auto rows = tuningRows(a);
    for (size_t i = 0; i < rows.size(); ++i) {
        int d = tuningMinus(int(i)).contains(click) ? -1 : tuningPlus(int(i)).contains(click) ? 1 : 0;
        if (!d) continue;
        rows[i].step(d);
        auto now = tuningRows(a);
        lightingMessage(a, rows[i].name + " set to " + now[i].value + " (saved)");
        return true;
    }
    return true;
}
// Identity of the current Fit + Save. The camera only moves between sessions (another group), so the
// allowance needs one hover check per new calibration, not per run.
string v13CalibrationId(const App& a) {
    if (!a.calibration || !a.calibration->valid) return "";
    const auto& h = a.calibration->mapping.h;
    ostringstream s;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) s << std::setprecision(7) << h(r, c) << ',';
    std::ostringstream id;
    id << std::hex << std::hash<string>{}(s.str());
    return id.str();
}
bool v13AllowanceChecked(const App& a) {
    auto id = v13CalibrationId(a);
    return !id.empty() && v13Allowance().checkedFor == id; // Offset approval belongs to this calibration only.
}
Point2d v13AllowanceAt(const App& a, Point2d p) {
    if (!a.calibration || a.calibration->mapping.robot.size() < 4) return { 0, 0 };
    const auto& r = a.calibration->mapping.robot;
    double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (auto q : r) {
        x0 = min(x0, q.x);
        x1 = max(x1, q.x);
        y0 = min(y0, q.y);
        y1 = max(y1, q.y);
    }
    const auto& g = v13Allowance();
    Point2d o(0, 0); // Start with no extra offset inside the taught area.
    if (p.x < x0)
        o.x = -min(g.cap, g.xMinus * (x0 - p.x));
    else if (p.x > x1)
        o.x = min(g.cap, g.xPlus * (p.x - x1));
    if (p.y < y0)
        o.y = -min(g.cap, g.yMinus * (y0 - p.y));
    else if (p.y > y1)
        o.y = min(g.cap, g.yPlus * (p.y - y1));
    return o;
}
// Camera position (robot mm) -> robot target: operator trim plus the outside-box allowance.
// V13.9 centre line, ALL colours (was BLUE only): the centre line is the P1-P9 centre Y. Left of it
// Y minus = gain x distance, right of it Y plus = gain x distance, each capped. A visible side face
// pulls the detected centre towards the centre line, so the target is pushed back away from it.
double v13CentreOffsetY(const App& a, Point2d p, int colour = -1) {
    if (!a.calibration || a.calibration->mapping.robot.empty()) return 0;
    double yc = 0;
    for (auto q : a.calibration->mapping.robot) yc += q.y;
    yc /= a.calibration->mapping.robot.size();
    const auto& g = v13Allowance();
    if (p.y < yc) return -min(g.cap, g.yMinusFor(colour) * (yc - p.y));
    if (p.y > yc) return min(g.cap, g.yPlusFor(colour) * (p.y - yc));
    return 0.0;
}
Point2d v13Correction(const App& a, Point2d camera, int colour) {
    return v13AllowanceAt(a, camera) + Point2d(0, v13CentreOffsetY(a, camera, colour));
}
Point2d v13Target(const App& a, Point2d camera, int colour = -1) {
    return camera + a.dynamicTrim + v13Correction(a, camera, colour); // Add manual trim and camera correction, in robot millimetres.
}
// Fit all four mapped corners together rather than rejecting one noisy edge.
constexpr double BLOCK_ALIGNMENT_LIMIT_DEG = 12.0;
double blockAlignmentError(const calibrationui::Detection& d, const calibrationui::H& h) {
    if (d.quad.size() != 4) return 90;
    vector<Point2f> corners;
    for (auto q : d.quad) {
        auto p = calibrationui::project(h, q);
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) return 90;
        corners.emplace_back(float(p.x), float(p.y));
    }
    auto box = minAreaRect(corners);
    double angle = fmod(abs(double(box.angle)), 90.0);
    return min(angle, 90 - angle);
}
using calibrationui::v11SquareScore;
using calibrationui::v11TopFace;
// ---- Side face guard (disabled, kept for reference)
// Perspective shows a cube's side faces in the same colour as its top. Whatever part the estimator
// calls the top, the TRUE top face always lies inside the whole same-colour outline. The fingers open
// along robot Y, so if that outline fits inside the opening (minus the 3 mm margin) around the
// target, neither finger can land on the block. A target that fails is re-centred on the outline's
// Y midpoint (worst-case error (span-35)/2); an outline longer than 43.5 mm in Y is refused.
constexpr double V13_SAFE_HALF_Y_MM = GRIPPER_INNER_MM * .5 - CLEARANCE_MARGIN_MM; // 21.75 mm
bool v13OutlineY(const Mat& frame, const calibrationui::ColourModel& cfg, const calibrationui::H& h,
    const calibrationui::Detection& o, double& lo, double& hi) {
    double x0, x1;
    return calibrationui::v13Outline(frame, cfg, h, o, x0, x1, lo, hi);
}
// DISABLED: measured on 19 Oct frames, all 19 targets it re-centred moved TOWARDS the dark (shadow) side,
// because the colour outline includes the lamp shadow. Kept for reference; pick targets are V12 faces.
constexpr bool V13_OUTLINE_GUARD = false;
calibrationui::Detection v13ConstrainY(const Mat& frame, const calibrationui::ColourModel& cfg,
    const calibrationui::H& h, calibrationui::Detection d) {
    if (!V13_OUTLINE_GUARD || !d.pickFaceVerified) return d;
    double lo, hi;
    if (!v13OutlineY(frame, cfg, h, d, lo, hi)) return d;
    d.blobMinY = lo;
    d.blobMaxY = hi;
    auto centre = calibrationui::project(h, d.centre);
    double need = max(hi - centre.y, centre.y - lo);
    if (need <= V13_SAFE_HALF_Y_MM) return d;
    if ((hi - lo) * .5 > V13_SAFE_HALF_Y_MM) {
        d.pickFaceVerified = false;
        return d;
    } // no safe Y exists
    Point2d shift(0, (lo + hi) * .5 - centre.y);
    auto inv = h.inv();
    for (auto& q : d.quad) {
        auto w = calibrationui::project(h, Point2d(q)) + shift;
        auto px = calibrationui::project(inv, w);
        q = Point(cvRound(px.x), cvRound(px.y));
    }
    d.box = boundingRect(d.quad);
    d.centre = calibrationui::project(inv, centre + shift);
    return d;
}
void v11RefinePickupFaces(App& a) {
    if (!a.calibration || !a.calibration->valid || a.calibration->freeze) return;
    auto& c = *a.calibration;
    for (auto& d : c.detections) {
        if (a.stackVision.configured && (d.box & a.stackVision.stack) == d.box) continue;
        try {
            d = calibrationui::v13PickFace(c.frame, c.activeColourModel(), c.mapping.h, d, true);
            d = v13ConstrainY(c.frame, c.activeColourModel(), c.mapping.h, d);
        }
        catch (...) {
            d.pickFaceVerified = false;
        }
    }
}
bool blockAligned(const calibrationui::Detection& d, const calibrationui::H& h) {
    return d.pickFaceVerified && blockAlignmentError(d, h) <= BLOCK_ALIGNMENT_LIMIT_DEG;
}

// ------------------------------------------------------------
// SECTION 8  STATIC BUILD AND TEST PAGE ACTIONS
// ------------------------------------------------------------
bool lockedPickupReady(const App& a) {
    if (!a.calibration || (a.dynamicStep != 1 && a.dynamicStep != 2) || a.dynamicToken < 1 || a.dynamicToken > 127)
        return false;
    const auto& c = *a.calibration;
    if (!c.valid || c.opt.source != "camera" || c.freeze || c.entering || !c.test || c.planning || c.comparisonView)
        return false;
    try {
        dynamicFields(a.dynamicToken, 2, a.dynamicXY);
    }
    catch (...) {
        return false;
    }
    // The operator verifies HOVER COMPLETE and alignment. The arm may hide the block.
    return true;
}
constexpr double STACK_SETTLE_S = 30.0; // robot needs about this long to pick, stack and go home (no ACK exists)
double buildWaitLeft(const App& a) {
    if (a.actionAt.time_since_epoch().count() == 0) return 0;
    double since = chrono::duration<double>(chrono::steady_clock::now() - a.actionAt).count();
    return max(0.0, a.actionSettle - since);
}
// TEST-window stack build: 5 = pick the SELECTED block (XY locked now) and stack it at S1 level n;
// 6 = unstack everything and put each block back where it was picked.
void buildAction(App& a, int request) {
    auto& c = *a.calibration;
    double left = buildWaitLeft(a);
    if (left > 0) {
        a.dynamicStatus =
            "WAIT " + to_string(int(ceil(left))) +
            " s: robot is still moving; a frame sent now would be lost. Wait for the pendant to show HOME";
        return;
    }
    if (a.dynamicStep != 0) {
        a.dynamicStatus = "Finish the hover/pick session first (3, then 4)";
        return;
    }
    int frameAction = 8;
    Point2d xy;
    if (request == 5) {
        if (a.buildCount >= 3) {
            a.dynamicStatus = "Stack is full (3/3): press 6 RETURN ALL";
            return;
        }
        if (!c.valid || c.opt.source != "camera" || c.freeze || c.entering || !c.test || c.planning ||
            c.comparisonView) {
            a.dynamicStatus = "Need LIVE Coordinate Test with a valid real-camera mapping";
            return;
        }
        if (!c.selectStable || c.detections.empty()) {
            a.dynamicStatus = "Click one stationary block first";
            return;
        }
        auto d = c.detections[c.selected];
        auto p = v13Target(a, calibrationui::project(c.mapping.h, d.centre), d.colour);
        if (!calibrationui::reachOk(p)) {
            a.dynamicStatus = "Block is outside the robot reach (X>=200 mm, 360-540 mm from the base) - move it closer";
            return;
        }
        for (size_t j = 0; j < c.detections.size(); ++j)
            if (int(j) != c.selected &&
                abs(calibrationui::project(c.mapping.h, c.detections[j].centre).x - p.x) < a.v10.gapX &&
                abs(calibrationui::project(c.mapping.h, c.detections[j].centre).y - p.y) < a.v10.gapY) {
                a.dynamicStatus = "Isolate selected block: need centre separation X47 OR Y61 mm";
                return;
            }
        if (!blockAligned(d, c.mapping.h)) {
            a.dynamicStatus = "Block fitted angle exceeds 12 degrees from robot axes";
            return;
        }
        xy = Point2d(cvRound(p.x * 10) / 10.0, cvRound(p.y * 10) / 10.0);
        frameAction = 5 + a.buildCount;
    }
    else {
        if (a.buildCount < 1) {
            a.dynamicStatus = "Nothing stacked yet";
            return;
        }
        xy = a.buildXY[a.buildCount - 1];
    }
    try {
        int token = int(chrono::steady_clock::now().time_since_epoch().count() % 127) + 1;
        auto fields = dynamicFields(token, frameAction, xy);
        struct Guard {
            bool& b;
            Guard(bool& v) : b(v) { b = true; }
            ~Guard() { b = false; }
        } guard(a.dynamicSending);
        if (!sendFrame(a, 232, fields, 233)) {
            a.dynamicStatus = "TX FAILED: robot state unknown. Check the pendant before any further action.";
            return;
        }
        a.actionAt = chrono::steady_clock::now();
        if (request == 5) {
            a.buildXY[a.buildCount] = xy;
            a.buildCount++;
            a.actionSettle = STACK_SETTLE_S;
            a.dynamicStatus = "BLOCK " + to_string(a.buildCount) +
                "/3 SENT (XY locked before motion): robot picks it and stacks at S1 level " +
                to_string(a.buildCount) + ". Wait for pendant STACKED LEVEL " + to_string(a.buildCount) +
                ", then select the next block";
        }
        else {
            a.actionSettle = 25.0 * a.buildCount + 15.0;
            a.buildCount = 0;
            a.dynamicStatus =
                "RETURN ALL SENT: robot unstacks top first and puts each block back where it was picked. Wait for pendant ALL BLOCKS RETURNED";
        }
        cout << a.dynamicStatus << " X=" << xy.x << " Y=" << xy.y << endl;
    }
    catch (const exception& e) {
        a.dynamicStatus = e.what();
    }
}
constexpr double HOVER_SETTLE_S =
25.0; // robot needs this long to reach hover; a PICK frame sent earlier is lost (no ACK)
double testCommandWait(const App& a) {
    auto stamp = a.lastTestTx.time_since_epoch().count() ? a.lastTestTx : a.hoverSentAt;
    if (!stamp.time_since_epoch().count()) return HOVER_SETTLE_S;
    return max(0.0, HOVER_SETTLE_S - chrono::duration<double>(chrono::steady_clock::now() - stamp).count());
}
bool supervisedPickReady(const App& a) {
    return lockedPickupReady(a) && v10HoverPoint(a, a.dynamicXY);
}
void dynamicAction(App& a, int action) {
    if (action >= 4) {
        a.dynamicStatus = "V10: use DYNAMIC for stacking; TEST is hover/pick/replace only";
        return;
    }
    if (a.busy || !a.calibrationMode || !a.dynamicTestMode || !a.calibration) return;
    auto& c = *a.calibration;
    if (a.lightCaptureActive) {
        a.dynamicStatus = "Finish/cancel lighting capture before Test motion";
        return;
    }
    if (action != 3 && !c.v11TopFaceCalibration) {
        a.dynamicStatus = "Top-face calibration required: redo Fit + Save and verify XY before Test motion.";
        return;
    }
    if (c.diagnosticRemaining) {
        a.dynamicStatus = "Detection capture active: PC motion blocked until capture finishes";
        return;
    }
    if (c.enlargedValidation) {
        a.dynamicStatus =
            "MEASURE AREA: robot commands disabled. Use pendant to measure actual XY, then RECORD ACTUAL XY.";
        return;
    }
    if (!a.serialConnected || a.sentCount) {
        a.dynamicStatus = "Blocked: COM offline or stack cycle active";
        return;
    }
    if (action == 5 || action == 6) {
        buildAction(a, action);
        return;
    }
    if (action == 7) { // operator saw that the last level did NOT get stacked: allow the same level to be sent again
        if (a.buildCount < 1) {
            a.dynamicStatus = "Nothing to retry";
            return;
        }
        a.buildCount--;
        a.actionAt = {};
        a.actionSettle = 0;
        a.dynamicStatus = "Count set to " + to_string(a.buildCount) +
            "/3. Select the block again and press 5 to redo level " + to_string(a.buildCount + 1) +
            " (check the stack by eye first)";
        return;
    }
    if (action == 1 || action == 2 || action == 4) {
        if (!c.valid || c.opt.source != "camera" || c.freeze || c.entering || !c.test || c.planning ||
            c.comparisonView) {
            a.dynamicStatus = "Need LIVE Coordinate Test with a valid real-camera mapping";
            return;
        }
    }
    if (action == 1 || action == 4) {
        if (a.dynamicStep != 0) {
            a.dynamicStatus = "Finish the active hover session first";
            return;
        }
        if (c.detections.empty() || c.selectionMissing) {
            a.dynamicStatus = "Selected block not visible: wait for a fresh detection or reselect if moved";
            return;
        }
        auto d = c.detections[c.selected];
        if (c.selectedStability.ready) d.centre = c.selectedStability.centre;
        auto xy = v13Target(a, calibrationui::project(c.mapping.h, d.centre), d.colour);
        xy = { cvRound(xy.x * 2) / 2.0, cvRound(xy.y * 2) / 2.0 };
        if (!v10HoverPoint(a, xy)) {
            a.dynamicStatus = "Hover blocked: " + v10HoverBlockReason(a, xy);
            return;
        }
        if (!c.selectStable) {
            a.dynamicStatus = "Selected block: " + c.selectedStability.reason;
            return;
        }
        for (size_t j = 0; j < c.detections.size(); ++j)
            if (int(j) != c.selected &&
                abs(calibrationui::project(c.mapping.h, c.detections[j].centre).x - xy.x) < a.v10.gapX &&
                abs(calibrationui::project(c.mapping.h, c.detections[j].centre).y - xy.y) < a.v10.gapY) {
                a.dynamicStatus = "Isolate selected block: need centre separation X47 OR Y61 mm";
                return;
            }
        // Fixed tool orientation: fitted top-face alignment, 12-degree tolerance.
        if (!blockAligned(d, c.mapping.h)) {
            a.dynamicStatus = "Block fitted angle exceeds 12 degrees from robot axes";
            return;
        }
        a.dynamicXY = Point2d(cvRound(xy.x * 10) / 10.0, cvRound(xy.y * 10) / 10.0);
        a.dynamicToken = int(chrono::steady_clock::now().time_since_epoch().count() % 127) + 1;
        a.dynamicColour = d.colour;
    }
    else if (action == 2) {
        if (!v10HoverPoint(a, a.dynamicXY)) {
            a.dynamicStatus = "Locked target outside current Test limits; use CANCEL + HOME";
            return;
        }
        if (!lockedPickupReady(a)) {
            a.dynamicStatus = "Need a valid locked HOVER session; verify pendant and alignment before PICK";
            return;
        }
        if (testCommandWait(a) > 0) {
            a.dynamicStatus =
                "WAIT " + to_string(int(ceil(testCommandWait(a)))) +
                " s before follow-up: pendant must show HOVER/PICK COMPLETE and READY. Serial has no queue/ACK.";
            return;
        }
        // Use immutable pre-hover XY/token; NEVER retarget from arm-occluded live detections.
    }
    else if (action == 3) {
        if (a.dynamicStep != 1 && a.dynamicStep != 2 && a.dynamicStep != 3) {
            a.dynamicStatus = "No active hover/pick session";
            return;
        }
        if (testCommandWait(a) > 0) {
            a.dynamicStatus =
                "WAIT " + to_string(int(ceil(testCommandWait(a)))) +
                " s: controller may still be moving. After pendant READY, resend the SAME Home/Replace if needed.";
            return;
        }
    }
    else
        return;
    try {
        auto fields = dynamicFields(a.dynamicToken, action, a.dynamicXY);
        struct Guard {
            bool& b;
            Guard(bool& v) : b(v) { b = true; }
            ~Guard() { b = false; }
        } guard(a.dynamicSending);
        cout << "TEST FRAME token=" << a.dynamicToken << " action=" << action << " locked XY=" << a.dynamicXY.x << ","
            << a.dynamicXY.y << " wire=";
        for (int j = 2; j < 6; ++j) cout << int(fields[j]) << (j == 5 ? '\n' : ',');
        bool ok = sendFrame(a, 232, fields, 233);
        if (!ok) {
            a.dynamicStatus = "TX FAILED: robot state unknown. Use controller stop/recovery; no automatic retry.";
            return;
        }
        a.lastTestTx = chrono::steady_clock::now();
        if (action == 1) a.hoverSentAt = a.lastTestTx;
        a.dynamicStep = action == 4 ? 2 : action;
        a.dynamicStatus =
            action == 4
            ? "GO + PICK SENT: XY was locked BEFORE motion. Watch the pendant for PICK/LIFT COMPLETE, then click REPLACE + HOME"
            : action == 1
            ? "HOVER SENT. Wait for pendant HOVER COMPLETE, check the gripper is over the block by eye, then click 2 CONFIRM ALIGNED: PICK"
            : action == 2
            ? "PICK SENT: verify pendant PICK/LIFT COMPLETE before REPLACE"
            : "HOME/REPLACE SENT (NO ACK): if no motion after pendant READY, use 3 RESEND SAME. Verify actual home/release before 4";
        if (action == 1 && !v10Point(a, a.dynamicXY))
            a.dynamicStatus =
            "ENLARGED HOVER SENT: verify pendant COMPLETE/READY, alignment and pickup Z before 2 CONFIRM ALIGNED: PICK. 3 CANCEL + HOME available.";
        cout << a.dynamicStatus << " X=" << a.dynamicXY.x << " Y=" << a.dynamicXY.y << endl;
    }
    catch (const exception& e) {
        a.dynamicStatus = e.what();
    }
}

void drawStack(Mat&, int, int, int);

// ------------------------------------------------------------
// SECTION 9  DYNAMIC MODE: WORKSPACE, PLANS AND PERMITS
//    Where blocks may be picked and returned, how a cycle is planned and encoded, and how the PC
//    releases the robot from each HOME checkpoint (permit) once the camera has checked the last transfer.
// ------------------------------------------------------------
bool v10PhysicalPoint(const App& a, Point2d p) {
    const auto& v = a.v10;
    if (!a.calibration || !a.calibration->valid || !isfinite(p.x) || !isfinite(p.y)) return false;
    if (p.x - 30 < v.minX || p.x + 30 > v.maxX || p.y - 40 < v.minY || p.y + 40 > v.maxY) return false;
    if (!calibrationui::reachOk(p)) return false;
    for (double x : {372.65, 442.65, 512.65})
        if (abs(p.x - x) < 60 && abs(p.y - 185.98) < 80) return false;
    return true;
}
bool v10BasePoint(const App& a, Point2d p) {
    return v10PhysicalPoint(a, p) &&
        (calibrationui::regionDistance(a.calibration->mapping, p) >= 2 || a.calibration->enlargedCoverage(p));
}
bool enlargedHoverPoint(const App& a, Point2d p) {
    if (!a.calibration || !a.calibration->valid || !isfinite(p.x) || !isfinite(p.y)) return false;
    if (p.x < a.hoverMinX || p.x > a.hoverMaxX || p.y < a.hoverMinY || p.y > a.hoverMaxY) return false;
    if (!calibrationui::reachOk(p)) return false;
    for (double x : {372.65, 442.65, 512.65})
        if (abs(p.x - x) < 60 && abs(p.y - 185.98) < 80) return false;
    return true;
}
bool v10HoverPoint(const App& a, Point2d p) {
    return a.enlargedHover ? enlargedHoverPoint(a, p) : v10Point(a, p);
}
string v10HoverBlockReason(const App& a, Point2d p) {
    if (!a.calibration || !a.calibration->valid) return "valid calibration required";
    if (!isfinite(p.x) || !isfinite(p.y)) return "non-finite XY";
    const auto& v = a.v10;
    const auto& c = *a.calibration;
    if (a.enlargedHover) {
        if (p.x < a.hoverMinX || p.x > a.hoverMaxX)
            return "X " + calibrationui::fmt(p.x, 1) + " outside HOVER CENTRE X " + calibrationui::fmt(a.hoverMinX, 1) +
            ".." + calibrationui::fmt(a.hoverMaxX, 1) + " mm";
        if (p.y < a.hoverMinY || p.y > a.hoverMaxY)
            return "Y " + calibrationui::fmt(p.y, 1) + " outside HOVER CENTRE Y " + calibrationui::fmt(a.hoverMinY, 1) +
            ".." + calibrationui::fmt(a.hoverMaxY, 1) + " mm";
    }
    else {
        if (calibrationui::regionDistance(c.mapping, p) < 2 && !c.enlargedCoverage(p))
            return "ENLARGED HOVER is OFF. Turn it ON to test outside yellow; workspace limits still apply";
        if (p.x - 30 < v.minX || p.x + 30 > v.maxX)
            return "X " + calibrationui::fmt(p.x, 1) + " outside allowed CENTRE X " +
            calibrationui::fmt(v.minX + 30, 1) + ".." + calibrationui::fmt(v.maxX - 30, 1) +
            " mm (edge margins included)";
        if (p.y - 40 < v.minY || p.y + 40 > v.maxY)
            return "Y " + calibrationui::fmt(p.y, 1) + " outside allowed CENTRE Y " +
            calibrationui::fmt(v.minY + 40, 1) + ".." + calibrationui::fmt(v.maxY - 40, 1) +
            " mm (edge margins included)";
    }
    if (!calibrationui::reachOk(p))
        return "numerical XY radius " + calibrationui::fmt(hypot(p.x, p.y), 1) +
        " mm outside 360..540, or X below 200; not measured robot reach";
    for (double x : {372.65, 442.65, 512.65})
        if (abs(p.x - x) < 60 && abs(p.y - 185.98) < 80) return "inside inflated stack exclusion zone";
    return "outside selected/validated area";
}
void drawEnlargedHover(Mat& ui, const App& a, bool dynamic = false) {
    if ((dynamic ? !a.v10.enlargedReturns : !a.enlargedHover) || !a.calibration || !a.calibration->valid ||
        a.calibration->frame.empty())
        return;
    const auto& c = *a.calibration;
    auto view = calibrationui::VIDEO;
    double scale = min(double(view.width) / c.frame.cols, double(view.height) / c.frame.rows);
    Point2d offset(view.x + (view.width - c.frame.cols * scale) / 2, view.y + (view.height - c.frame.rows * scale) / 2);
    auto inverse = c.mapping.h.inv();
    auto screen = [&](Point2d xy) {
        auto p = calibrationui::project(inverse, xy);
        return Point(cvRound(p.x * scale + offset.x), cvRound(p.y * scale + offset.y));
        };
    static Mat regionMask;
    static calibrationui::H cachedMapping = calibrationui::H::zeros();
    static Size cachedSize;
    static array<double, 4> cachedBounds{ {0, 0, 0, 0} };
    array<double, 4> bounds{ {a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY} };
    if (regionMask.empty() || cachedSize != c.frame.size() || cachedBounds != bounds ||
        norm(Mat(cachedMapping) - Mat(c.mapping.h)) > 1e-8) {
        regionMask = Mat::zeros(view.size(), CV_8U);
        cachedMapping = c.mapping.h;
        cachedSize = c.frame.size();
        cachedBounds = bounds;
        for (double x = a.hoverMinX; x < a.hoverMaxX; x += 10)
            for (double y = a.hoverMinY; y < a.hoverMaxY; y += 10) {
                double right = min(x + 10, a.hoverMaxX), top = min(y + 10, a.hoverMaxY);
                if (!enlargedHoverPoint(a, { (x + right) / 2, (y + top) / 2 })) continue;
                vector<Point> q;
                for (auto p : vector<Point2d>{ {x, y}, {right, y}, {right, top}, {x, top} })
                    q.push_back(screen(p) - view.tl());
                fillConvexPoly(regionMask, q, Scalar(255));
            }
    }
    Mat base = ui(view), tinted;
    addWeighted(base, .84, Mat(base.size(), base.type(), Scalar(160, 70, 150)), .16, 0, tinted);
    tinted.copyTo(base, regionMask);
    vector<Point> border;
    for (auto p : vector<Point2d>{ {a.hoverMinX, a.hoverMinY},
                                  {a.hoverMaxX, a.hoverMinY},
                                  {a.hoverMaxX, a.hoverMaxY},
                                  {a.hoverMinX, a.hoverMaxY} })
        border.push_back(screen(p) - view.tl());
    Mat roi = ui(view);
    polylines(roi, border, true, Scalar(220, 100, 210), 2);
    label(ui, dynamic ? "Purple: working area" : "Purple: hover area   Yellow: calibration", { 22, 122 }, .40, GOLD);
    if (!dynamic)
        label(ui,
            string("Hover X ") + calibrationui::fmt(a.hoverMinX, 0) + ".." + calibrationui::fmt(a.hoverMaxX, 0) +
            " Y " + calibrationui::fmt(a.hoverMinY, 0) + ".." + calibrationui::fmt(a.hoverMaxY, 0) +
            " / radius 360..540mm",
            { 22, 143 }, .38, GOLD);
}
bool v10OriginalPoint(Point2d p) {
    const auto refs = calibrationui::nominalRobot();
    vector<Point2f> polygon;
    for (int i : {0, 2, 8, 6}) polygon.push_back(Point2f(float(refs[i].x), float(refs[i].y)));
    return pointPolygonTest(polygon, p, true) >= 2;
}
bool v10Point(const App& a, Point2d p) {
    if (a.dynamicOperate && a.v10.enlargedReturns) return enlargedHoverPoint(a, p);
    return v10BasePoint(a, p) && (!a.dynamicOperate || v10OriginalPoint(p));
}
// V13.8: a pickup is judged where the block IS (camera position, as drawn in the purple area). The
// offsets (outside-box allowance, centre-line Y) can move the robot target a few mm beyond the rectangle
// edge; that target only has to stay reachable and outside the exclusions, as the MOD itself checks.
bool v13TargetSafe(const App& a, Point2d t) {
    if (!(a.dynamicOperate && a.v10.enlargedReturns)) return v10Point(a, t);
    if (!a.calibration || !a.calibration->valid || !isfinite(t.x) || !isfinite(t.y)) return false;
    double m = 2 * v13Allowance().cap + max(abs(a.dynamicTrim.x), abs(a.dynamicTrim.y)) + 1;
    if (t.x < a.hoverMinX - m || t.x > a.hoverMaxX + m || t.y < a.hoverMinY - m || t.y > a.hoverMaxY + m) return false;
    if (!calibrationui::reachOk(t)) return false;
    for (double x : {372.65, 442.65, 512.65})
        if (abs(t.x - x) < 60 && abs(t.y - 185.98) < 80) return false;
    return true;
}
bool v13PickInside(const App& a, Point2d camera, Point2d target) {
    return v10Point(a, camera) && v13TargetSafe(a, target);
}
bool v10Conflict(Point2d a, Point2d b, const V10State& v) {
    return abs(a.x - b.x) < v.gapX && abs(a.y - b.y) < v.gapY;
}
// ---- Release tracking: what the scene looked like when the robot was sent
double v13Seconds(chrono::steady_clock::time_point later, chrono::steady_clock::time_point earlier) {
    return earlier.time_since_epoch().count() ? chrono::duration<double>(later - earlier).count() : 0.0;
}
bool v13InStack(const App& a, const calibrationui::Detection& d) {
    return a.stackVision.configured && (d.box & a.stackVision.stack) == d.box;
}
vector<SceneItem> v13Scene(const App& a, bool excludeStack) {
    vector<SceneItem> out;
    if (!a.calibration) return out;
    const auto& c = *a.calibration;
    for (const auto& d : c.detections) {
        if (excludeStack && v13InStack(a, d)) continue;
        out.push_back({ d.colour, calibrationui::project(c.mapping.h, d.centre) });
    }
    return out;
}
void v13CompensatedDelta(const Mat& reference, const Mat& frame, Scalar shift, Mat& delta);
// Baseline of the stack zone for checkpoint `step`. The robot is parked at HOME when released, so the
// CURRENT frame's own workspace check (not the 0.5 s debounced flag) decides; otherwise the next
// clear frame is used, provided the arm has not appeared yet.
void v13TakeBase(App& a, int step) {
    auto& r = a.v10.v13;
    const auto& sv = a.stackVision;
    if (step < 1 || step > 6 || !r.basePending[step] || !a.calibration || !sv.configured || sv.clearFrames < 1) return;
    const Mat& f = a.calibration->frame;
    if (f.empty() || (sv.stack & Rect(0, 0, f.cols, f.rows)) != sv.stack) return;
    r.stackBase[step] = f(sv.stack).clone();
    r.basePending[step] = false;
}
// The robot has been permitted to move and will next wait at checkpoint `step` (0 = no checkpoint).
void v13Release(App& a, int step, chrono::steady_clock::time_point now) {
    auto& r = a.v10.v13;
    r.releasedStep = step;
    r.releasedAt = now;
    r.armSeen = false;
    r.clearSince = {};
    r.stillSince = {};
    r.anchorScene.clear();
    r.mismatch = 0;
    r.evidenceKind = 0;
    r.evidenceSince = {};
    r.samples.clear();
    r.sampleTimes.clear();
    r.sampleKey = -1;
    r.lastProgress = now;
    r.nextSearchSince = {};
    r.snapshot = v13Scene(a, true);
    r.releaseStart = a.lastStart;
    r.releaseEnd = a.lastEnd;
    r.releaseFields = a.lastFields;
    r.releaseResends = 0;
    if (step >= 1 && step <= 6) {
        r.deltaOK[step] = false;
        r.stackBase[step].release();
        r.basePending[step] = true;
        v13TakeBase(a, step);
    }
}
// Before/after comparison of the stack zone for one checkpoint. Perspective makes the side faces of a
// tall stack visible, which can break the square-top detector; the CHANGE since the robot left does
// not depend on the shape: one connected patch, mostly one colour, about one face in size.
struct V13Delta {
    bool valid = false;
    int area = 0, coloured = 0, colour = -1, baseColoured = 0, baseColour = -1, blobs = 0;
    double share = 0, baseShare = 0;
    bool border = false;
};
V13Delta v13StackDelta(const App& a, int step) {
    V13Delta out;
    const auto& r = a.v10.v13;
    const auto& sv = a.stackVision;
    if (step < 1 || step > 6 || !a.calibration || r.stackBase[step].empty() || !sv.configured) return out;
    const Mat& frame = a.calibration->frame;
    if (frame.empty() || (sv.stack & Rect(0, 0, frame.cols, frame.rows)) != sv.stack ||
        r.stackBase[step].size() != sv.stack.size())
        return out;
    Mat cur = frame(sv.stack), base = r.stackBase[step];
    vector<int> d[3];
    for (int y = 0; y < cur.rows; y += 2)
        for (int x = 0; x < cur.cols; x += 2) {
            auto b = base.at<Vec3b>(y, x), c = cur.at<Vec3b>(y, x);
            int bc = max(b[0], max(b[1], b[2])) - min(b[0], min(b[1], b[2])),
                cc = max(c[0], max(c[1], c[2])) - min(c[0], min(c[1], c[2]));
            if (bc > 40 || cc > 40 || abs(int(c[0]) - int(b[0])) > 45 || abs(int(c[1]) - int(b[1])) > 45 ||
                abs(int(c[2]) - int(b[2])) > 45)
                continue;
            for (int k = 0; k < 3; ++k) d[k].push_back(int(c[k]) - int(b[k]));
        }
    Scalar shift(0, 0, 0, 0);
    if (d[0].size() >= 50)
        for (int k = 0; k < 3; ++k) {
            nth_element(d[k].begin(), d[k].begin() + d[k].size() / 2, d[k].end());
            shift[k] = d[k][d[k].size() / 2];
        }
    Mat delta, m;
    v13CompensatedDelta(base, cur, shift, delta);
    vector<Mat> ch;
    split(delta, ch);
    max(ch[0], ch[1], m);
    max(m, ch[2], m);
    threshold(m, m, 40, 255, THRESH_BINARY);
    morphologyEx(m, m, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, { 3, 3 }));
    morphologyEx(m, m, MORPH_CLOSE, getStructuringElement(MORPH_RECT, { 5, 5 }));
    out.area = countNonZero(m);
    auto model = stackColourModel(a);
    model.staticStyle = false;
    model.greenRescue = false;
    model.sat = min(model.sat, 55);
    model.val = min(model.val, 35);
    const int defaults[3] = { 0, 60, 110 }, tolerance[3] = { 13, 25, 20 };
    for (int k = 0; k < 3; ++k)
        if (model.hue[k] == defaults[k]) model.tol[k] = max(model.tol[k], tolerance[k]);
    Mat colouredNow = Mat::zeros(m.size(), CV_8U);
    auto classify = [&](const Mat& image, int& colour, double& share, Mat* keep) {
        Mat hsv;
        cvtColor(image, hsv, COLOR_BGR2HSV);
        array<int, 3> n{};
        for (int k = 0; k < 3; ++k) {
            Mat cm = calibrationui::colourThreshold(hsv, k, model);
            bitwise_and(cm, m, cm);
            n[k] = countNonZero(cm);
            if (keep) *keep |= cm;
        }
        int sum = n[0] + n[1] + n[2];
        colour = int(max_element(n.begin(), n.end()) - n.begin());
        share = sum ? double(n[colour]) / sum : 0;
        if (!sum) colour = -1;
        return sum;
        };
    out.coloured = classify(cur, out.colour, out.share, &colouredNow);
    out.baseColoured = classify(base, out.baseColour, out.baseShare, nullptr);
    Mat labels, stats, centres;
    int parts = connectedComponentsWithStats(colouredNow, labels, stats, centres, 8, CV_32S);
    for (int i = 1; i < parts; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) < sv.faceArea * .15) continue;
        ++out.blobs;
        int x = stats.at<int>(i, CC_STAT_LEFT), y = stats.at<int>(i, CC_STAT_TOP), w = stats.at<int>(i, CC_STAT_WIDTH),
            h = stats.at<int>(i, CC_STAT_HEIGHT);
        if (x <= 0 || y <= 0 || x + w >= m.cols || y + h >= m.rows) out.border = true;
    }
    out.valid = true;
    return out;
}
bool v13DeltaAdded(const V13Delta& d, int colour, double face) {
    return d.valid && d.coloured >= .35 * face && d.colour == colour && d.share >= .6 && d.blobs == 1 && !d.border;
}
bool v13DeltaNothing(const V13Delta& d, double face) {
    return d.valid && d.area < .25 * face;
}
bool v13DeltaSomething(const V13Delta& d, double face) {
    return d.valid && d.area >= .6 * face;
}
// The block that WAS on top has gone and the lower layer (or the empty site) is what changed into view.
bool v13DeltaRemoved(const V13Delta& d, int removed, int below, double face) {
    if (!d.valid || d.area < .35 * face || d.baseColour != removed || d.baseShare < .6 || d.border || d.blobs > 1)
        return false;
    return below < 0 ? d.coloured < .2 * face : (d.coloured < .2 * face || (d.colour == below && d.share >= .5));
}
vector<uint8_t> v10Fields(int token, int pattern, const array<Point2d, 3>& points, bool returning) {
    vector<uint8_t> out{ uint8_t(token) };
    if (token < 1 || token > 127 || pattern < 1 || pattern > 3) throw runtime_error("Invalid V10 token/pattern");
    if (!returning) out.push_back(uint8_t(pattern)); // Only a stack plan needs the colour pattern.
    for (auto p : points) {
        auto one = dynamicFields(token, 1, p);
        out.insert(out.end(), one.begin() + 2, one.begin() + 6); // Copy just the four encoded XY bytes for this block.
    }
    int sum = 0;
    for (auto b : out) sum += b;
    out.push_back(uint8_t(sum % 251 + 1)); // Keep zero free for the separators between wire values.
    return out;
}
void v10Journal(const App& a) {
    FileStorage f("v10_cycle_state.yml", FileStorage::WRITE); // Record the cycle so an unfinished run is noticed after a restart.
    if (!f.isOpened()) throw runtime_error("Cannot save V10 cycle journal; transmission blocked");
    f << "active" << int(a.v10.phase != 0) << "phase" << a.v10.phase << "token" << a.v10.token;
    for (int i = 0; i < 3; ++i)
        f << ("pick" + to_string(i)) << a.v10.pick[i] << ("drop" + to_string(i)) << a.v10.drop[i];
}
void v10Send(App& a, bool returning) {
    auto& v = a.v10;
    auto fields = v10Fields(v.token, a.selected, returning ? v.drop : v.pick, returning);
    struct Guard {
        bool& b;
        Guard(bool& x) : b(x) { b = true; }
        ~Guard() { b = false; }
    } guard(a.dynamicSending);
    if (!sendFrame(a, returning ? 222 : 220, fields, returning ? 223 : 221))
        throw runtime_error("TX failed - robot state unknown; controller recovery required");
    if (returning) v.lastReturnTx = chrono::steady_clock::now(); // Start the return resend timer after sending the frame.
    v.visionLast = {};
    v.visionPeriod = 0;
    if (!returning || a.v10.v13.releasedStep == 0) v13Release(a, returning ? 4 : 1, chrono::steady_clock::now());
}
array<double, 4> v10ActiveBounds(const App& a);
// Margin between a planned return (block centre) and ANY limit of the working area.
constexpr double V10_RETURN_INSET_MM = 30;
// Returning safely inside the region also improves the following camera acquisition.
bool v10ReturnVisible(const App& a, Point2d p) {
    // V13.6: 30 mm, not 15. The camera reading of a returned block differs from the commanded drop by the
    // pick offsets (up to 8 mm) plus mapping error outside the P1-P9 tray (4-9 mm) plus release slip, so a
    // 15 mm inset could leave the next pickup reading just outside the workspace rectangle.
    auto b = v10ActiveBounds(a);
    const double inset = V10_RETURN_INSET_MM;
    if (p.x < b[0] + inset || p.x > b[1] - inset || p.y < b[2] + inset || p.y > b[3] - inset) return false;
    const auto& c = *a.calibration;
    if (c.frame.empty()) return false;
    vector<Point2f> corners;
    for (auto delta : vector<Point2d>{ {-22, -22}, {22, -22}, {22, 22}, {-22, 22} }) {
        auto q = calibrationui::project(c.mapping.h.inv(), p + delta);
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || q.x < 20 || q.y < 20 || q.x >= c.frame.cols - 20 ||
            q.y >= c.frame.rows - 20)
            return false;
        corners.emplace_back(float(q.x), float(q.y));
    }
    auto face = minAreaRect(corners);
    double small = min(face.size.width, face.size.height), large = max(face.size.width, face.size.height);
    if (small < 14 || large > small * 2.0) return false;
    Rect roi = boundingRect(corners);
    Mat gray;
    cvtColor(c.frame(roi), gray, COLOR_BGR2GRAY);
    return mean(gray)[0] >= 45;
}
// Robust global exposure change (lamp ON/OFF) between the saved empty background and this frame,
// measured on neutral (low-chroma) pixels only so coloured blocks and the arm do not bias it.
Scalar v13ExposureShift(const Mat& reference, const Mat& frame) {
    if (reference.empty() || frame.empty() || reference.size() != frame.size() || reference.type() != CV_8UC3 ||
        frame.type() != CV_8UC3)
        return Scalar(0, 0, 0, 0);
    vector<int> d[3];
    for (int y = 4; y < frame.rows; y += 8)
        for (int x = 4; x < frame.cols; x += 8) {
            auto r = reference.at<Vec3b>(y, x), f = frame.at<Vec3b>(y, x);
            int rc = max(r[0], max(r[1], r[2])) - min(r[0], min(r[1], r[2])),
                fc = max(f[0], max(f[1], f[2])) - min(f[0], min(f[1], f[2]));
            if (rc > 40 || fc > 40 || r[2] < 25) continue;
            for (int k = 0; k < 3; ++k) d[k].push_back(int(f[k]) - int(r[k]));
        }
    Scalar shift(0, 0, 0, 0);
    if (d[0].size() < 300) return shift;
    for (int k = 0; k < 3; ++k) {
        nth_element(d[k].begin(), d[k].begin() + d[k].size() / 2, d[k].end());
        shift[k] = max(-45, min(45, d[k][d[k].size() / 2]));
    }
    return shift;
}
// absdiff after removing the global exposure shift; a local block or arm still shows as change.
void v13CompensatedDelta(const Mat& reference, const Mat& frame, Scalar shift, Mat& delta) {
    Mat adjusted;
    add(reference, shift, adjusted);
    absdiff(adjusted, frame, delta);
}
bool v10Empty(const App& a, Point2d p) {
    const auto& c = *a.calibration;
    const auto& ref = a.v10.emptyBackground;
    if (ref.empty() || ref.size() != c.frame.size()) return false;
    vector<Point2d> corners;
    for (auto d : vector<Point2d>{ {-25, -39}, {25, -39}, {25, 39}, {-25, 39} })
        corners.push_back(calibrationui::project(c.mapping.h.inv(), p + d));
    vector<Point> pixels;
    for (auto q : corners) pixels.push_back(Point(cvRound(q.x), cvRound(q.y)));
    Rect roi = boundingRect(pixels);
    if ((roi & Rect(0, 0, ref.cols, ref.rows)) != roi || roi.area() < 16) return false;
    Mat delta, gray, mask;
    v13CompensatedDelta(ref(roi), c.frame(roi), a.v10.exposureShift, delta);
    cvtColor(delta, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 3);
    threshold(gray, mask, 30, 255, THRESH_BINARY);
    return double(countNonZero(mask)) / mask.total() < .06;
}
// `correction` = deliberate V13 offsets (left allowance + centre-line Y). The face is shifted by it first,
// so a deliberate correction is not counted as aiming error. Operator trim is still counted, as in V12.
bool v11GripperOpeningFits(const calibrationui::Detection& d, const calibrationui::H& h, Point2d target,
    Point2d correction = { 0, 0 }) {
    if (!d.pickFaceVerified || d.quad.size() != 4) return false;
    // Opening runs along robot Y. Keep the measured3mm allowance inside both fingers.
    double halfOpening = GRIPPER_INNER_MM * .5 - CLEARANCE_MARGIN_MM;
    // A smaller painted patch does not shrink the physical35mm cube.
    auto centre = calibrationui::project(h, calibrationui::crossCentre(d.quad)) + correction,
        q0 = calibrationui::project(h, d.quad[0]);
    auto u = calibrationui::project(h, d.quad[1]) - q0, v = calibrationui::project(h, d.quad[3]) - q0;
    if (norm(u) < 1 || norm(v) < 1) return false;
    double physicalHalfY = 17.5 * (abs(u.y) / norm(u) + abs(v.y) / norm(v));
    if (!isfinite(centre.y) || abs(centre.y - target.y) + physicalHalfY > halfOpening) return false;
    for (auto px : d.quad) {
        auto q = calibrationui::project(h, px) + correction;
        if (!isfinite(q.y) || abs(q.y - target.y) > halfOpening) return false;
    }
    // V13: the whole same-colour outline (top + visible side) must also fit between the fingers.
    if (isfinite(d.blobMinY) && isfinite(d.blobMaxY) &&
        (d.blobMaxY - target.y > halfOpening || target.y - d.blobMinY > halfOpening))
        return false;
    return true;
}
// Chooses one block of each colour for the selected pattern. Each must be inside the working area,
// square to the gripper, clear of its neighbours and reachable after the offsets.
bool v10PickPlan(App& a, array<Point2d, 3>& xy, array<int, 3>& colours, string* why = nullptr) {
    auto& c = *a.calibration;
    const auto& pattern = calibrationui::PATTERNS[a.selected - 1];
    array<vector<Point2d>, 3> options;
    for (int level = 0; level < 3; ++level) {
        int wanted = pattern[2 - level];
        colours[level] = wanted;
        int seen = 0, inside = 0, unused = 0, alignedCount = 0, openingCount = 0;
        string clearanceDetail, alignDetail, openingDetail;
        for (const auto& d : c.detections) {
            if (d.colour != wanted) continue;
            ++seen;
            auto p = v13Target(a, calibrationui::project(c.mapping.h, d.centre), d.colour);
            p = { cvRound(p.x * 2) / 2.0, cvRound(p.y * 2) / 2.0 };
            if (!v13PickInside(a, calibrationui::project(c.mapping.h, d.centre), p)) continue;
            ++inside;
            if (!a.v10.allowRepeats &&
                any_of(a.v10.used.begin(), a.v10.used.end(), [&](Point2d q) { return norm(p - q) < 25; }))
                continue;
            ++unused;
            bool aligned = blockAligned(d, c.mapping.h);
            if (!aligned) { // say WHICH check failed: ambiguous top face or rotation
                if (!d.pickFaceVerified)
                    alignDetail =
                    "top face not found (side face merged or block outline not block sized). Turn or nudge the block slightly";
                else
                    alignDetail = "block rotated " + calibrationui::fmt(blockAlignmentError(d, c.mapping.h), 0) +
                    " deg (limit " + calibrationui::fmt(BLOCK_ALIGNMENT_LIMIT_DEG, 0) +
                    "). Straighten it to the gripper";
                continue;
            }
            ++alignedCount;
            if (!v11GripperOpeningFits(d, c.mapping.h, p,
                v13Correction(a, calibrationui::project(c.mapping.h, d.centre), d.colour))) {
                auto corr = v13Correction(a, calibrationui::project(c.mapping.h, d.centre), d.colour);
                auto centre = calibrationui::project(c.mapping.h, calibrationui::crossCentre(d.quad)) + corr;
                openingDetail = " (aim error Y " + calibrationui::fmt(abs(centre.y - p.y), 1) + " mm incl. trim " +
                    calibrationui::fmt(a.dynamicTrim.y, 1) + "; needs <= " +
                    calibrationui::fmt(GRIPPER_INNER_MM * .5 - CLEARANCE_MARGIN_MM - 17.5, 2) +
                    " mm; face edges " +
                    calibrationui::fmt(norm(calibrationui::project(c.mapping.h, d.quad[1]) -
                        calibrationui::project(c.mapping.h, d.quad[0])),
                        1) +
                    "/" +
                    calibrationui::fmt(norm(calibrationui::project(c.mapping.h, d.quad[3]) -
                        calibrationui::project(c.mapping.h, d.quad[0])),
                        1) +
                    " mm)";
                continue;
            }
            ++openingCount;
            bool nearOther = false;
            for (const auto& other : c.detections) {
                if (&other == &d) continue;
                auto occupied = calibrationui::project(c.mapping.h, other.centre);
                if (v10Conflict(p, occupied, a.v10)) {
                    nearOther = true;
                    clearanceDetail = string(calibrationui::COLOUR_NAMES[wanted]) + " near " +
                        calibrationui::COLOUR_NAMES[other.colour] + ": X" +
                        calibrationui::fmt(abs(p.x - occupied.x), 1) + " Y" +
                        calibrationui::fmt(abs(p.y - occupied.y), 1) +
                        " mm. Need X>=" + calibrationui::fmt(a.v10.gapX, 0) +
                        " OR Y>=" + calibrationui::fmt(a.v10.gapY, 0) + " mm";
                    break;
                }
            }
            if (!nearOther) options[level].push_back(p);
        }
        if (options[level].empty()) {
            if (why) {
                string colour = calibrationui::COLOUR_NAMES[wanted];
                *why = !seen ? "Cannot see a " + colour + " block"
                    : !inside ? "Move " + colour + " inside the selected workspace"
                    : !unused ? colour + " already used. Select ALLOW REPEATS to run the returned blocks again"
                    : !alignedCount ? colour + ": " + alignDetail
                    : !openingCount ? "Gripper opening cannot clear the " + colour + " face" + openingDetail
                    : (clearanceDetail.empty() ? "Separate nearby block centres for clearance"
                        : clearanceDetail);
            }
            return false;
        }
        if (!a.v10.planHistory.empty())
            stable_sort(options[level].begin(), options[level].end(), [&](Point2d x, Point2d y) {
            return norm(x - a.v10.anchor[level]) < norm(y - a.v10.anchor[level]);
                });
    }
    for (auto p : options[0])
        for (auto q : options[1])
            for (auto r : options[2])
                if (!v10Conflict(p, q, a.v10) && !v10Conflict(p, r, a.v10) && !v10Conflict(q, r, a.v10)) {
                    xy = { {p, q, r} };
                    if (why) why->clear();
                    return true;
                }
    if (why) *why = "Move the blocks further apart for gripper clearance";
    return false;
}
bool v10SelectPattern(App& a, int pattern) {
    auto& v = a.v10;
    if (pattern < 1 || pattern > 3 || a.busy || v.phase == 5) return false;
    if (v.phase) {
        v.nextPattern = pattern;
        return true;
    }
    if (a.selected != pattern) {
        v.startRequested = false;
        // A pattern only changes RGB pickup order. Reorder the existing evidence by colour.
        auto old = calibrationui::PATTERNS[a.selected - 1], next = calibrationui::PATTERNS[pattern - 1];
        auto reorder = [&](array<Point2d, 3> points) {
            auto result = points;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (next[2 - i] == old[2 - j]) result[i] = points[j];
            return result;
            };
        v.anchor = reorder(v.anchor);
        v.pendingAnchor = reorder(v.pendingAnchor);
        for (auto& sample : v.planHistory) sample = reorder(sample);
        a.selected = pattern;
    }
    return true;
}
void v10SetRepeatMode(App& a, bool repeat) {
    auto& v = a.v10;
    if (v.phase || a.busy) return;
    v.startRequested = false;
    v.allowRepeats = repeat;
    // Current eligibility/position checks still run before Start; unchanged blocks retain readings.
    v.status = repeat ? "Dynamic ALLOW REPEATS: returned blocks can be used in the next cycle"
        : "Dynamic UNUSED ONLY: returned blocks excluded until New Batch";
}
void v10ResetReadings(App& a) {
    auto& v = a.v10;
    if (v.phase || a.busy) return;
    v.startRequested = false;
    v.planHistory.clear();
    v.planTimes.clear();
    v.planOutliers = 0;
    v.stableFrames = 0;
    if (a.calibration) {
        a.calibration->detectionFilter.clear();
        a.calibration->selectedStability.clear();
    }
    v.status = "Readings reset. Calibration and background kept.";
}
void v10UpdatePickupSamples(V10State& v, bool valid, const array<Point2d, 3>& p, chrono::steady_clock::time_point now) {
    if (v.planHistory.empty()) {
        v.planTimes.clear();
        v.planOutliers = 0;
    }
    // Expire only after a real gap in accepted evidence, not by total collection duration.
    // A slow but continuously valid camera must be able to collect five samples.
    if (!v.planTimes.empty() && chrono::duration<double>(now - v.planTimes.back()).count() > 1.5) {
        v.planHistory.clear();
        v.planTimes.clear();
        v.planOutliers = 0;
    }
    // Retain progress during loss. Readiness separately requires a current valid plan and recent sample.
    if (!valid) {
        v.stableFrames = int(v.planHistory.size());
        return;
    }
    if (!v.planHistory.empty()) {
        double jump = 0;
        for (int i = 0; i < 3; ++i) jump = max(jump, norm(p[i] - v.anchor[i]));
        if (jump > 5) {
            bool same = v.planOutliers > 0;
            for (int i = 0; i < 3; ++i)
                if (norm(p[i] - v.pendingAnchor[i]) > 3) same = false;
            if (same)
                ++v.planOutliers;
            else {
                v.pendingAnchor = p;
                v.planOutliers = 1;
            }
            if (v.planOutliers < 3) {
                v.stableFrames = int(v.planHistory.size());
                return;
            }
            v.planHistory.clear();
            v.planTimes.clear();
            v.stableFrames = 0;
            v.planOutliers = 0;
        }
        else
            v.planOutliers = 0;
    }
    if (v.planHistory.empty()) v.anchor = p; // Use the first accepted positions to spot later movement.
    // Do not replace the five-sample window faster than its minimum settling span.
    // Otherwise a fast (>33fps) camera can show5/5 forever without satisfying120ms.
    if (!v.planTimes.empty() && chrono::duration<double>(now - v.planTimes.back()).count() < .035) {
        v.stableFrames = int(v.planHistory.size());
        return;
    }
    v.planHistory.push_back(p);
    v.planTimes.push_back(now);
    while (v.planHistory.size() > 20) {
        v.planHistory.pop_front();
        v.planTimes.pop_front();
    }
    v.stableFrames = int(v.planHistory.size());
}
bool v10PickupSamplesReady(const V10State& v, const array<Point2d, 3>& current, chrono::steady_clock::time_point now) {
    if (v.planHistory.size() < 5 || v.planTimes.size() != v.planHistory.size() ||
        chrono::duration<double>(now - v.planTimes.back()).count() > .75)
        return false;
    // Use the most recent five real images, with a minimum time span and bounded age.
    auto first = v.planTimes[v.planTimes.size() - 5];
    if (chrono::duration<double>(v.planTimes.back() - first).count() < .12) return false;
    for (int i = 0; i < 3; ++i)
        if (norm(current[i] - v.anchor[i]) > 5) return false;
    return true;
}
bool v10MedianPickup(const V10State& v, array<Point2d, 3>& target) {
    if (v.planHistory.size() < 5) return false;
    for (int i = 0; i < 3; ++i) {
        vector<double> x, y;
        for (size_t n = v.planHistory.size() - 5; n < v.planHistory.size(); ++n) {
            x.push_back(v.planHistory[n][i].x);
            y.push_back(v.planHistory[n][i].y);
        }
        sort(x.begin(), x.end());
        sort(y.begin(), y.end());
        target[i] = { cvRound(x[2] * 2) / 2.0, cvRound(y[2] * 2) / 2.0 }; // Use the middle of five readings, rounded to 0.5 mm wire steps.
    }
    return true;
}
string v10ReadyReason(App& a) {
    auto& v = a.v10;
    auto& c = *a.calibration;
    if (a.lightCaptureActive) return "Finish/cancel lighting capture before starting a cycle";
    if (!c.valid) return "Open Calibrate and save the camera mapping";
    if (!c.v11TopFaceCalibration)
        return "Top-face estimator changed: redo Fit + Save with35mm cubes, then verify independent XY positions";
    if (c.opt.source != "camera" || c.freeze) return "A live camera view is required";
    if (v.v13.frozen) return "Camera image frozen (identical frames); reconnect the camera";
    if (c.enlargedValidation) return "Exit Measure Area in Test first";
    if (v.emptyBackground.empty() || !v.workspaceChecked)
        return "Open CALIBRATE and capture clear reference with all blocks and arm removed";
    if (!a.serialConnected) return "Robot connection offline. Start the program with its COM port";
    auto visual = stackStartReason(a);
    if (!visual.empty()) return visual;
    array<Point2d, 3> p;
    array<int, 3> col;
    string why;
    if (!v10PickPlan(a, p, col, &why)) return why;
    if (!v10PickupSamplesReady(v, p, chrono::steady_clock::now())) {
        if (v.planHistory.size() < 5)
            return "Reading block positions: " + to_string(min(5, v.stableFrames)) +
            "/5. Matching readings retained through brief gaps";
        if (v.planTimes.size() != v.planHistory.size() ||
            chrono::duration<double>(chrono::steady_clock::now() - v.planTimes.back()).count() > .75)
            return "Five readings collected; waiting for a fresh matching camera frame";
        if (chrono::duration<double>(v.planTimes.back() - v.planTimes[v.planTimes.size() - 5]).count() < .12)
            return "Five readings collected; verifying position stability";
        return "Five readings collected; position jitter or changed source: waiting for matching positions";
    }
    return "";
}
bool v10ReturnAreaOK(const App& a, Point2d p) {
    return a.v10.enlargedReturns ? enlargedHoverPoint(a, p) : (v10BasePoint(a, p) && v10OriginalPoint(p));
}
// 24 Oct: the 30 mm return margin used to apply to the rectangle edges only. In the Enlarged area the
// working area is also limited by robot reach (360..540 mm from the base, X >= 200) and the stack-site
// exclusions, so a return could land right on the 540 mm arc and then read as "(outside)" and never be
// picked again. Now every point within 30 mm of the drop must be inside the working area as well.
bool v10ReturnAreaInset(const App& a, Point2d p) {
    if (!v10ReturnAreaOK(a, p)) return false;
    for (int k = 0; k < 16; ++k) {
        double angle = k * CV_PI / 8;
        if (!v10ReturnAreaOK(a, p + V10_RETURN_INSET_MM * Point2d(cos(angle), sin(angle)))) return false;
    }
    // Exact for the reach arcs: the nearest arc points lie on the line through the robot base.
    double r = norm(p);
    if (r < 1) return false;
    Point2d radial = p * (V10_RETURN_INSET_MM / r);
    return v10ReturnAreaOK(a, p + radial) && v10ReturnAreaOK(a, p - radial);
}
array<double, 4> v10ActiveBounds(const App& a) {
    return a.v10.enlargedReturns ? array<double, 4>{{a.hoverMinX, a.hoverMaxX, a.hoverMinY, a.hoverMaxY}}
    : array<double, 4>{ {a.v10.minX, a.v10.maxX, a.v10.minY, a.v10.maxY} };
}
void v10SelectReturnArea(App& a, bool enlarged) {
    if (a.v10.phase || a.busy) return;
    a.v10.startRequested = false;
    if (a.v10.enlargedReturns != enlarged && !(a.stackVision.savedReference && a.stackVision.configured)) {
        a.v10.emptyBackground.release();
        a.v10.workspaceChecked = false;
    }
    a.v10.enlargedReturns = enlarged;
    if (a.stackVision.savedReference && a.stackVision.configured) a.v10.backgroundRange = v10ActiveBounds(a);
    a.v10.stableFrames = 0;
    a.v10.planHistory.clear();
    a.v10.status =
        enlarged
        ? "ENLARGED pickup + return: operator-tested range selected. Saved clear reference is reused when valid"
        : "ORIGINAL pickup + return: original P1-P9 centre region within valid calibration";
}
// Picks three free, visible return positions at random (clear of every block, 30 mm inside the area).
bool v10ReturnPlan(App& a) {
    auto& v = a.v10;
    vector<Point2d> candidates;
    vector<Point2d> occupied;
    // Reserve measurement/settling margin so next-cycle pickup does not sit on the clearance threshold.
    auto returnConflict = [&](Point2d p, Point2d q) {
        return abs(p.x - q.x) < v.gapX + 10 && abs(p.y - q.y) < v.gapY + 10;
        };
    for (auto& d : a.calibration->detections)
        occupied.push_back(calibrationui::project(a.calibration->mapping.h, d.centre));
    auto bounds = v10ActiveBounds(a);
    for (double x = ceil(bounds[0] / 5) * 5; x <= bounds[1]; x += 5)
        for (double y = ceil(bounds[2] / 5) * 5; y <= bounds[3]; y += 5) {
            Point2d p(x, y);
            if (!v10ReturnAreaInset(a, p) || !v10ReturnVisible(a, p)) continue;
            if (any_of(occupied.begin(), occupied.end(), [&](Point2d q) { return returnConflict(p, q); })) continue;
            if (v10Empty(a, p)) candidates.push_back(p);
        }
    for (int attempt = 0; attempt < 30; ++attempt) {
        shuffle(candidates.begin(), candidates.end(), a.rng);
        int n = 0;
        for (auto p : candidates) {
            bool good = true;
            for (int j = 0; j < n; ++j)
                if (returnConflict(p, v.drop[j])) good = false;
            if (good) v.drop[n++] = p;
            if (n == 3) return true;
        }
    }
    return false;
}
void v10PauseForRecovery(App& a) {
    auto& v = a.v10;
    v.startRequested = false;
    v.autoRun = false;
    v.phase = 5;
    v.recoveryArmed = false;
    v.quietSince = {};
    v.planHistory.clear();
    v.planTimes.clear();
    v.stableFrames = 0;
    v.status = "PC paused (the robot is not stopped). For a fault use the robot STOP, then RECOVERY HELP.";
}
vector<uint8_t> v11ControlFields(int token, int action, int step) {
    if (token < 1 || token > 127 || action < 1 || action > 4 || step < 1 || step > 6)
        throw runtime_error("V11 invalid control fields");
    return { uint8_t(token), uint8_t(action), uint8_t(step), uint8_t((token + action + step) % 251 + 1) };
}
void v11SendControl(App& a, int action, int step) {
    auto fields = v11ControlFields(a.v10.token, action, step);
    bool prior = a.dynamicSending;
    a.dynamicSending = true;
    bool ok = sendFrame(a, 234, fields, 235);
    a.dynamicSending = prior;
    if (!ok) throw runtime_error("V11 control TX failed; controller state unknown");
    a.v10.controlLast = chrono::steady_clock::now();
    a.v10.visionLast = {};
    a.v10.visionPeriod = 0;
    // A permit releases the robot towards the next checkpoint. Resends do not restart the clock.
    if (action == 4) {
        int next = (step == 3 || step == 6) ? 0 : step + 1;
        if (a.v10.v13.releasedStep != next) v13Release(a, next, a.v10.controlLast);
    }
}
// V13 frame 236..237: token, mode(1 re-pick, 2 retarget+permit), step 1..6, attempt 1..127, X/Y (4 bytes, 0.5 mm), checksum.
vector<uint8_t> v13TargetFields(int token, int mode, int step, int attempt, Point2d xy) {
    if (token < 1 || token > 127 || mode < 1 || mode > 2 || step < 1 || step > 6 || attempt < 1 || attempt > 127)
        throw runtime_error("V13 invalid re-pick fields");
    if (mode == 2 && (step == 3 || step == 6)) throw runtime_error("V13 retarget has no next transfer");
    auto one = dynamicFields(token, 1, xy);
    vector<uint8_t> out{ uint8_t(token), uint8_t(mode), uint8_t(step), uint8_t(attempt) };
    out.insert(out.end(), one.begin() + 2, one.begin() + 6); // Copy just the four encoded XY bytes for this block.
    int sum = 0;
    for (auto b : out) sum += b;
    out.push_back(uint8_t(sum % 251 + 1)); // Keep zero free for the separators between wire values.
    return out;
}
// `fresh` = a new attempt (restarts the settling clock); false = identical resend.
void v13SendTarget(App& a, int mode, int step, int attempt, Point2d xy, bool fresh,
    chrono::steady_clock::time_point tick = chrono::steady_clock::now()) {
    auto fields = v13TargetFields(a.v10.token, mode, step, attempt, xy);
    bool prior = a.dynamicSending;
    a.dynamicSending = true;
    bool ok = sendFrame(a, 236, fields, 237);
    a.dynamicSending = prior;
    if (!ok) throw runtime_error("V13 re-pick TX failed; controller state unknown");
    // The robot can only move after the whole frame arrived: time the release from the END of TX.
    auto now = max(tick, chrono::steady_clock::now());
    a.v10.controlLast = now;
    a.v10.visionLast = {};
    a.v10.visionPeriod = 0;
    if (fresh) {
        int next = mode == 1 ? step : step + 1;
        v13Release(a, next, now);
    }
}
void v11OperatorRequest(App& a, int action) {
    if (a.lightCaptureActive) cancelLightingCapture(a, "Lighting capture cancelled for recovery request.");
    auto& v = a.v10;
    v.startRequested = false;
    if (a.sentCount || a.dynamicStep || a.buildCount || (!a.dynamicOperate && a.busy)) {
        a.message = a.dynamicStatus = v.status =
            "Static/Test recovery requires pendant STOP and physical reconciliation; Dynamic checkpoint controls cannot interrupt that session.";
        return;
    }
    if (action != 3) {
        v.resetArmed = v.resetAwait = v.resetPacketSent = false;
    }
    if (action == 3) {
        if (v.resetAwait) {
            if (a.busy || !v.resetPacketSent) {
                v.status = "Wait for the RESET packet, then check pendant RESET COMPLETE before confirming.";
                return;
            }
            v.operatorRequest = v.retryControl = 0;
            v.resetAwait = v.resetArmed = false;
            v.pumpFailed = false;
            v.phase = 0;
            v.v13 = V13State{};
            v.autoRun = false;
            v.nextPattern = 0;
            v.returnObserved = false;
            v.returnConfirmed.fill(false);
            v.planHistory.clear();
            v.planTimes.clear();
            v.stableFrames = 0;
            v.used.clear();
            for (auto& e : v.returnEvidence) e.clear();
            for (auto& e : v.sourceEvidence) e.clear();
            v.handshakeStep = 1;
            a.sentCount = 0;
            a.dynamicStep = 0;
            a.dynamicRequest = 0;
            a.buildCount = 0;
            a.commandLatched = false;
            a.pendingControllerMode = 0;
            a.stackVision.begin({ {0, 1, 2} });
            a.stackVision.armed = false;
            a.stackVision.resetFresh();
            try {
                v10Journal(a);
            }
            catch (const exception& e) {
                v.phase = 5;
                v.status = e.what();
                return;
            }
            a.message = a.dynamicStatus = v.status =
                "PC reset after operator-confirmed controller RESET COMPLETE. No motion or gripper command.";
            return;
        }
        if (!v.resetArmed) {
            v.resetArmed = true;
            v.operatorRequest = 1;
            v.retryControl = 1;
            v.phase = 5;
            v.status =
                "RESET STEP1: HOLD requested. Recover block/gripper on pendant. Click RESET: RECOVERED only after physical recovery.";
            return;
        }
        v.resetAwait = true;
        v.resetArmed = false;
        v.resetPacketSent = false;
    }
    v.operatorRequest = action;
    v.retryControl = action;
    v.phase = 5;
    v.status =
        action == 1 ? "HOLD requested: acts at next taught approach checkpoint; NOT a physical safety stop."
        : action == 2
        ? "HOME requested: controller only permits it at a known approach checkpoint. Check pendant; no gripper release."
        : "Controller RESET requested after operator recovery. Wait for V11 CONTROLLER RESET COMPLETE, then CONFIRM PC RESET.";
    a.message = a.dynamicStatus = v.status;
    try {
        v10Journal(a);
    }
    catch (const exception& e) {
        v.status = e.what();
    }
}
void v11ServiceOperator(App& a) {
    auto& v = a.v10;
    if (a.busy || !v.retryControl) return;
    auto now = chrono::steady_clock::now();
    if (!v.operatorRequest && v.controlLast.time_since_epoch().count() &&
        chrono::duration<double>(now - v.controlLast).count() < 15)
        return;
    int action = v.retryControl;
    v.operatorRequest = 0;
    if (v.token < 1) {
        if (action == 1 || action == 3) {
            v.token = 1;
            try {
                v10Journal(a);
            }
            catch (const exception& e) {
                v.retryControl = 0;
                v.status = e.what();
                return;
            }
        }
        else {
            v.retryControl = 0;
            v.status =
                "No matching Dynamic token for HOME. Recover on pendant; use acknowledged RESET to reconcile state.";
            return;
        }
    }
    try {
        v11SendControl(a, action, max(1, min(6, v.handshakeStep)));
        if (action == 3) v.resetPacketSent = true;
    }
    catch (const exception& e) {
        v.status = e.what();
        v.retryControl = 0;
    }
}
bool v11PickupEmptyNow(const App& a, int index) {
    const auto& c = *a.calibration;
    auto target = a.v10.pick[index];
    for (auto d : c.detections)
        if (norm(calibrationui::project(c.mapping.h, d.centre) - target) < 25) return false;
    if (a.v10.emptyBackground.empty() || a.v10.emptyBackground.size() != c.frame.size()) return false;
    vector<Point> q;
    auto inverse = c.mapping.h.inv();
    for (auto off : vector<Point2d>{ {-11, -11}, {11, -11}, {11, 11}, {-11, 11} }) {
        auto p = calibrationui::project(inverse, target + off);
        q.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    Rect patch = boundingRect(q);
    if ((patch & Rect(0, 0, c.frame.cols, c.frame.rows)) != patch || patch.area() < 25) return false;
    Mat delta, gray, mask;
    v13CompensatedDelta(a.v10.emptyBackground(patch), c.frame(patch), a.v10.exposureShift, delta);
    cvtColor(delta, gray, COLOR_BGR2GRAY);
    threshold(gray, mask, 35, 255, THRESH_BINARY);
    return double(countNonZero(mask)) / mask.total() < .06;
}
bool v12SourceAbsent(const App& a, int index) {
    const auto& c = *a.calibration;
    // V13: a different-colour neighbour knocked next to the source is not the picked block.
    for (auto d : c.detections)
        if (d.colour == a.v10.colours[index] &&
            norm(calibrationui::project(c.mapping.h, d.centre) - a.v10.pick[index]) < 25)
            return false;
    return true;
}
bool v12PlacementClear(const App& a, int step, chrono::steady_clock::time_point now) {
    const auto& vision = a.stackVision;
    const auto& c = *a.calibration;
    return step >= 1 && step <= 3 && !c.freeze && vision.fresh(now) && vision.progress >= step &&
        vision.progress <= 3 && vision.clearStable &&
        ((vision.topClean && vision.top == vision.expected[vision.progress - 1]) || a.v10.v13.deltaOK[step]) &&
        v12SourceAbsent(a, step - 1);
}
bool v12PickupVerified(const App& a, int index, chrono::steady_clock::time_point now) {
    return v11PickupEmptyNow(a, index) || v12PlacementClear(a, index + 1, now);
}
string v12StackWaitReason(const App& a, int level, chrono::steady_clock::time_point now) {
    const auto& vision = a.stackVision;
    if (!vision.fresh(now)) return "waiting for a new camera frame";
    if (vision.progress < level) return "waiting for the expected colour at this stack level";
    bool byChange = a.v10.v13.deltaOK[level];
    if (!vision.topClean && !byChange)
        return "layer seen; confirming the stack once the robot is parked and the view is still (" + vision.note + ")";
    if (!byChange && (vision.progress > 3 || vision.top != vision.expected[vision.progress - 1]))
        return "current top differs from the recorded stacking sequence";
    if (!v12SourceAbsent(a, level - 1)) return "a block is still visible at the original pickup position";
    if (!v12PickupVerified(a, level - 1, now))
        return "pickup background differs and the independent clear-placement check is not ready";
    if (!a.v10.sourceEvidence[level - 1].ready(now)) return "collecting fresh pickup-clear confirmations (4 required)";
    return "placement verified; continue permission is retried until the robot advances";
}
int v13NextPickDecision(App& a, int step, chrono::steady_clock::time_point now, Point2d& xy, bool& fresh, string& why);
// Stacking checkpoints: when layer `step` is verified, permit the robot (or move the next target).
void v11StackHandshake(App& a, chrono::steady_clock::time_point now) {
    auto& v = a.v10;
    for (int i = 0; i < 3; ++i) v.sourceEvidence[i].update(v12PickupVerified(a, i, now), now);
    if (v.handshakeStep < 1 || v.handshakeStep > 3) return;
    int step = v.handshakeStep;
    if (step < 3 && a.stackVision.progress > step) {
        ++v.handshakeStep;
        v.controlLast = {};
        v.handshakeSince = now;
        step = v.handshakeStep;
    }
    // progress is 0 until layer 1 is seen: check the range before indexing expected[progress - 1].
    bool layerOK = (a.stackVision.progress >= 1 && a.stackVision.progress <= 3 && a.stackVision.topClean &&
        a.stackVision.top == a.stackVision.expected[a.stackVision.progress - 1]) ||
        v.v13.deltaOK[step];
    bool passed = a.stackVision.progress >= step && a.stackVision.progress <= 3 && layerOK &&
        v12PickupVerified(a, step - 1, now) && v.sourceEvidence[step - 1].ready(now) &&
        a.stackVision.fresh(now);
    if (passed &&
        (!v.controlLast.time_since_epoch().count() || chrono::duration<double>(now - v.controlLast).count() > 2.5)) {
        // V13: the camera checks the NEXT block before releasing the robot towards it.
        Point2d next;
        bool fresh = false;
        string why;
        int mode = step < 3 ? v13NextPickDecision(a, step, now, next, fresh, why) : 0;
        if (mode < 0) {
            v.v13.hint = "Layer " + to_string(step) + " verified. " + why;
            if (!v.handshakeSince.time_since_epoch().count()) v.handshakeSince = now;
            return;
        }
        if (mode == 2) {
            v13SendTarget(a, 2, step, v.v13.retargetAttempt[step], next, fresh, now);
            v.v13.permitKind[step] = 2;
            if (v.phase == 5) return;
            v.handshakeSince = chrono::steady_clock::now();
            v.status = "Layer " + to_string(step) + " OK. Next block moved, new pickup X" +
                calibrationui::fmt(next.x, 1) + " Y" + calibrationui::fmt(next.y, 1);
        }
        else {
            v11SendControl(a, 4, step);
            v.v13.permitKind[step] = 1;
            if (v.phase == 5) return;
            v.handshakeSince = chrono::steady_clock::now();
            v.status = "Layer " + to_string(step) + " OK. Robot continues.";
        }
    }
    // A vision delay is a retryable checkpoint wait, not a latched recovery HOLD.
    // Wrong/missing blocks still receive no permission; operator HOLD remains available.
    if (!v.handshakeSince.time_since_epoch().count()) v.handshakeSince = now;
}
void v10Action(App& a, int action) {
    auto& v = a.v10;
    auto& c = *a.calibration;
    if (a.busy) return;
    if (action != 3) v.startRequested = false;
    try {
        if (action == 8 && v.phase >= 1 && v.phase <= 4) {
            v11OperatorRequest(a, 1);
            return;
        }
        if (action == 7 && v.phase == 0) {
            if (!c.valid || c.frame.empty() || c.opt.source != "camera" || c.freeze) {
                v.status = "Preparation needs a saved calibration and live camera";
                return;
            }
            for (auto& d : c.detections)
                if (enlargedHoverPoint(a, calibrationui::project(c.mapping.h, d.centre)) ||
                    v10BasePoint(a, calibrationui::project(c.mapping.h, d.centre))) {
                    v.status = "Remove the blocks and arm, then press Prepare again";
                    return;
                }
            // Clicking Prepare acknowledges the physical workspace check and captures only an empty view.
            v.workspaceChecked = true;
            v10Action(a, 2);
            v.setupOpen = false;
            return;
        }
        if (action == 1 && v.phase == 0) {
            v.workspaceChecked = !v.workspaceChecked;
            v.status = v.workspaceChecked ? "Operator marked physical workspace/exclusions checked"
                : "Workspace check cleared";
            return;
        }
        if (action == 2 && v.phase == 0) {
            if (!c.valid || c.frame.empty() || c.freeze || c.opt.source != "camera" || !v.workspaceChecked) {
                v.status = "Need real calibration and physical workspace check";
                return;
            }
            for (auto& d : c.detections)
                if (enlargedHoverPoint(a, calibrationui::project(c.mapping.h, d.centre)) ||
                    v10BasePoint(a, calibrationui::project(c.mapping.h, d.centre))) {
                    v.status = "Remove blocks/arm from the workspace before EMPTY capture";
                    return;
                }
            v.emptyBackground = c.frame.clone();
            v.backgroundMapping = c.mapping.h;
            v.backgroundRange = v10ActiveBounds(a);
            v.status = "Empty background captured. Place blocks without moving camera/support.";
            return;
        }
        if (action == 3 && v.phase == 0) {
            auto reason = v10ReadyReason(a);
            if (!reason.empty()) {
                v.status = reason;
                return;
            }
            if (!v10PickPlan(a, v.pick, v.colours)) {
                v.status = "Block positions changed. Please wait for a fresh reading";
                return;
            }
            if (!v10MedianPickup(v, v.pick)) {
                v.status = "Need five recent pickup-plan samples";
                return;
            }
            for (auto p : v.pick)
                if (!v13TargetSafe(a, p)) {
                    v.status = "Median target outside workspace";
                    return;
                }
            array<Point2d, 3> latest;
            array<int, 3> latestColours;
            if (!v10PickPlan(a, latest, latestColours) ||
                !v10PickupSamplesReady(v, latest, chrono::steady_clock::now())) {
                v.status = "Current blocks changed; waiting for fresh valid positions";
                return;
            }
            for (int i = 0; i < 3; ++i)
                if (norm(v.pick[i] - latest[i]) > 5) {
                    v.status = "Block moved away from sampled pickup; waiting for fresh positions";
                    return;
                }
            for (int i = 0; i < 3; ++i)
                for (int j = i + 1; j < 3; ++j)
                    if (v10Conflict(v.pick[i], v.pick[j], v)) {
                        v.status = "Median pickup targets too close";
                        return;
                    }
            // The median command must clear every current unselected object too.
            for (int i = 0; i < 3; ++i)
                for (const auto& d : c.detections) {
                    auto occupied = calibrationui::project(c.mapping.h, d.centre);
                    if (d.colour == v.colours[i] && norm(v13Target(a, occupied, d.colour) - latest[i]) < 10) {
                        if (!v11GripperOpeningFits(d, c.mapping.h, v.pick[i], v13Correction(a, occupied, d.colour))) {
                            v.status =
                                "Start blocked: median pickup target leaves insufficient opening clearance for " +
                                string(calibrationui::COLOUR_NAMES[d.colour]);
                            return;
                        }
                        continue;
                    }
                    if (v10Conflict(v.pick[i], occupied, v)) {
                        v.status = "Sampled target too near another block. Separate blocks and refresh readings";
                        return;
                    }
                }
            a.stackVision.begin(v.colours);
            v.handshakeStep = 1;
            v.handshakeSince = {};
            v.controlLast = {};
            v.operatorRequest = v.retryControl = 0;
            v.resetArmed = v.resetAwait = false;
            for (auto& e : v.sourceEvidence) e.clear();
            v.token = v.token % 127 + 1;
            for (auto& evidence : v.returnEvidence) evidence.clear();
            v.returnConfirmed.fill(false);
            v.nextPattern = 0;
            v.returnObserved = false;
            v.finishAfter = {};
            v.quietSince = {};
            v.setupOpen = false;
            v.v13 = V13State{};
            v.phase = 1;
            v.cycleStarted = chrono::steady_clock::now();
            v10Journal(a);
            v10Send(a, false);
            v.status =
                "STACK PLAN SENT. Watching expected bottom-to-top colours, pickup removal and workspace-clear view before return.";
            return;
        }
        if (action == 5 && v.phase == 0) {
            v.used.clear();
            v.status = "New dynamic batch: handled blocks eligible again";
            return;
        }
        if (action == 6 && v.phase == 5) {
            v11OperatorRequest(a, 3);
            return;
        }
    }
    catch (const exception& e) {
        v.phase = 5;
        v.status = e.what();
        try {
            v10Journal(a);
        }
        catch (...) {}
    }
}
void v10CompleteByCamera(App& a) {
    auto& v = a.v10;
    for (auto p : v.drop) v.used.push_back(p);
    v.phase = 0;
    v.stableFrames = 0;
    v.planHistory.clear();
    v.planTimes.clear();
    v.quietSince = {};
    ++a.completedCycles;
    v.completedAt = chrono::steady_clock::now();
    if (v.nextPattern) {
        a.selected = v.nextPattern;
        v.nextPattern = 0;
    }
    v.status = string(v.autoRun ? "Cycle complete by camera. AUTO RUN starts the next cycle when Ready."
        : "Cycle complete by camera. Select S1/S2/S3 and Start when Ready.") +
        (v.v13.repicks || v.v13.retargets || v.v13.accepted
            ? " Camera handled " + to_string(v.v13.repicks) + " re-pick(s), " + to_string(v.v13.retargets) +
            " retarget(s), " + to_string(v.v13.accepted) + " relocated return(s)."
            : string(""));
}
constexpr double DYNAMIC_FINISH_PAUSE_S =
0.0; // No blind pause: fresh StackVision clear/empty evidence gates completion.
constexpr double RETURN_MATCH_MM = 15.0;
struct ReturnMatch {
    bool seen = false;
    double nearest = 1e9;
    int candidates = 0;
};
ReturnMatch v10ReturnMatch(const App& a, int i) {
    const auto& c = *a.calibration;
    const auto& v = a.v10;
    ReturnMatch result;
    auto inspect = [&](const vector<calibrationui::Detection>& objects, Point offset) {
        for (auto d : objects) {
            if (d.colour != v.colours[i]) continue;
            d.centre += Point2d(offset.x, offset.y);
            double gap = norm(v13Target(a, calibrationui::project(c.mapping.h, d.centre), d.colour) - v.drop[i]);
            result.nearest = min(result.nearest, gap);
            if (gap < RETURN_MATCH_MM) ++result.candidates;
        }
        };
    inspect(c.detections, {});
    if (result.candidates == 0 && !c.frame.empty() && c.valid) {
        // Search only the commanded destination, including after tracker expiry.
        // Every candidate must be a real square contour in this current image.
        vector<Point> corners;
        auto inverse = c.mapping.h.inv();
        for (auto off : vector<Point2d>{ {-42, -42}, {42, -42}, {42, 42}, {-42, 42} }) {
            auto px = calibrationui::project(inverse, v.drop[i] + off);
            if (!isfinite(px.x) || !isfinite(px.y) || abs(px.x) > 100000 || abs(px.y) > 100000) return result;
            corners.emplace_back(cvRound(px.x), cvRound(px.y));
        }
        Rect roi = boundingRect(corners) & Rect(0, 0, c.frame.cols, c.frame.rows);
        if (roi.width > 8 && roi.height > 8) {
            auto model = c.activeColourModel();
            model.staticStyle = false;
            model.greenRescue = true;
            model.minArea = max(80, min(model.minArea, 100));
            model.sat = min(model.sat, 55);
            model.val = min(model.val, 35);
            auto local = calibrationui::detect(c.frame(roi), model);
            vector<calibrationui::Detection> sized;
            for (auto d : local) {
                vector<Point2f> world;
                for (auto q : d.quad) {
                    auto p = calibrationui::project(c.mapping.h, Point2d(q.x + roi.x, q.y + roi.y));
                    world.emplace_back(float(p.x), float(p.y));
                }
                double area = abs(contourArea(world));
                if (area >= BLOCK_MM * BLOCK_MM * .40 && area <= BLOCK_MM * BLOCK_MM * 2.5) sized.push_back(d);
            }
            inspect(sized, roi.tl());
        }
    }
    result.seen = result.candidates == 1;
    return result;
}
int v10ReturnedCount(const App& a) {
    int count = 0;
    for (int i = 0; i < 3; ++i)
        if (v10ReturnMatch(a, i).seen) ++count;
    return count;
}
void v10ObserveReturns(App& a, chrono::steady_clock::time_point now) {
    string detail;
    for (int i = 0; i < 3; ++i) {
        auto match = v10ReturnMatch(a, i);
        a.v10.currentReturns[i] = match.seen;
        auto& evidence = a.v10.returnEvidence[i];
        evidence.update(match.seen, now);
        if (evidence.ready(now)) a.v10.returnConfirmed[i] = true;
        if (!evidence.ready(now)) {
            if (!detail.empty()) detail += "; ";
            detail += calibrationui::COLOUR_NAMES[a.v10.colours[i]];
            if (match.candidates > 1)
                detail += ": ambiguous";
            else if (match.seen)
                detail +=
                ": samples " + to_string(count(evidence.samples.begin(), evidence.samples.end(), true)) + "/4";
            else
                detail += match.nearest < 1e8 ? ": nearest " + calibrationui::fmt(match.nearest, 1) + "mm (need <15)"
                : ": not detected at return target";
        }
    }
    a.v10.returnDetail = detail;
}
int v10ConfirmedReturns(const App& a, chrono::steady_clock::time_point now) {
    int n = 0;
    for (int i = 0; i < 3; ++i)
        if (a.v10.returnEvidence[i].ready(now) || a.v10.v13.lost[i]) ++n;
    return n;
}
void v10RequestStart(App& a) {
    if (a.busy || a.v10.phase) return;
    a.v10.startRequested = true;
    a.v10.startDeadline = chrono::steady_clock::now() + chrono::seconds(2);
    a.v10.status = "START REQUESTED: checking the next live image. No robot command sent yet.";
}
bool v10FinishReady(const App& a, chrono::steady_clock::time_point now) {
    return a.v10.phase == 4 && a.v10.finishAfter.time_since_epoch().count() && now >= a.v10.finishAfter &&
        v10ConfirmedReturns(a, now) == 3 && a.stackVision.finishReady(now);
}

bool v10RetryAreaClear(const App& a) {
    for (auto p : a.v10.drop) {
        if (!v10ReturnAreaOK(a, p) || !v10ReturnVisible(a, p) || !v10Empty(a, p)) return false;
        for (const auto& d : a.calibration->detections)
            if (v10Conflict(p, calibrationui::project(a.calibration->mapping.h, d.centre), a.v10)) return false;
    }
    return true;
}
// ---- Supervisor helpers
// The PC has no robot feedback. A checkpoint is judged only when the robot has been released
// long enough for a whole transfer, the arm has left the view (it parks at HOME outside the
// camera) and the block positions have been still. Then the camera classifies what happened.

// ------------------------------------------------------------
// SECTION 10  DYNAMIC MODE: CAMERA SUPERVISOR
//    After every transfer the robot waits at HOME. Once the scene has settled (arm seen and gone, view
//    still), the stack zone is compared with how it looked when the robot left:
//      layer added as expected  -> permit the next transfer
//      nothing changed          -> arm never seen: resend the same command; arm seen: re-pick the block
//      next block was bumped    -> send its new position with the permit
//      stack knocked, wrong colour, block lost, 3 misses -> HOLD (operator needed)
// ------------------------------------------------------------
bool v13SceneMatches(const vector<SceneItem>& x, const vector<SceneItem>& y) {
    if (x.size() != y.size()) return false;
    vector<bool> used(y.size(), false);
    for (const auto& item : x) {
        bool found = false;
        for (size_t j = 0; j < y.size(); ++j)
            if (!used[j] && y[j].colour == item.colour && norm(y[j].xy - item.xy) <= V13_STILL_MM) {
                used[j] = found = true;
                break;
            }
        if (!found) return false;
    }
    return true;
}
void v13Observe(App& a, chrono::steady_clock::time_point now) {
    auto& r = a.v10.v13;
    auto scene = v13Scene(a, false);
    // Compare with the scene at the start of the still period. A one or two frame detection
    // flicker does not restart it; a persistent change (arm, sliding block) does.
    if (!r.stillSince.time_since_epoch().count()) {
        r.anchorScene = scene;
        r.stillSince = now;
        r.mismatch = 0;
    }
    else if (v13SceneMatches(scene, r.anchorScene))
        r.mismatch = 0;
    else if (++r.mismatch >= 3) {
        r.anchorScene = scene;
        r.stillSince = now;
        r.mismatch = 0;
    }
    bool clear = a.stackVision.fresh(now) && a.stackVision.clearStable;
    if (clear) {
        if (!r.clearSince.time_since_epoch().count()) r.clearSince = now;
    }
    else
        r.clearSince = {};
    // Arm = this frame is actually obstructed (not merely the 0.5 s debounce still counting).
    bool blocked = a.stackVision.fresh(now) && a.stackVision.clearFrames == 0;
    if (r.releasedStep && !r.armSeen && r.basePending[r.releasedStep] && !blocked) v13TakeBase(a, r.releasedStep);
    if (blocked && r.releasedStep) {
        r.armSeen = true;
        r.basePending[r.releasedStep] = false;
    }
}
// True once the robot has finished the transfer: arm seen and gone, view clear and still.
bool v13Settled(const V10State& v, chrono::steady_clock::time_point now) {
    const auto& r = v.v13;
    if (!r.releasedStep || !r.releasedAt.time_since_epoch().count()) return false;
    double t = v13Seconds(now, r.releasedAt);
    if (t < V13_MIN_TRANSFER_S) return false;
    double clear = v13Seconds(now, r.clearSince), still = v13Seconds(now, r.stillSince);
    if (clear >= V13_SETTLE_S && still >= V13_SETTLE_S && (r.armSeen || t >= V13_ARMLESS_S)) return true;
    // Workspace monitor unavailable (for example a lighting change): rely on a longer stillness.
    return still >= V13_FALLBACK_STILL_S && t >= V13_FALLBACK_AFTER_S;
}
struct V13Candidate {
    bool ok = false;
    Point2d xy;
    string why;
    int seen = 0;
};
// Same eligibility as v10PickPlan for one colour; the nearest eligible block to `near` wins.
V13Candidate v13FindPick(const App& a, int colour, Point2d near, const function<bool(Point2d)>& allowed = nullptr) {
    const auto& c = *a.calibration;
    V13Candidate best;
    double bestGap = 1e9;
    int seen = 0, inside = 0, unused = 0, aligned = 0, opening = 0;
    string clearance;
    string name = calibrationui::COLOUR_NAMES[colour];
    for (const auto& d : c.detections) {
        if (d.colour != colour || v13InStack(a, d)) continue;
        auto p = v13Target(a, calibrationui::project(c.mapping.h, d.centre), d.colour);
        p = { cvRound(p.x * 2) / 2.0, cvRound(p.y * 2) / 2.0 };
        if (allowed && !allowed(p)) continue;
        ++seen;
        if (!v13PickInside(a, calibrationui::project(c.mapping.h, d.centre), p)) continue;
        ++inside;
        if (!a.v10.allowRepeats &&
            any_of(a.v10.used.begin(), a.v10.used.end(), [&](Point2d q) { return norm(p - q) < 25; }))
            continue;
        ++unused;
        if (!blockAligned(d, c.mapping.h)) continue;
        ++aligned;
        if (!v11GripperOpeningFits(d, c.mapping.h, p,
            v13Correction(a, calibrationui::project(c.mapping.h, d.centre), d.colour)))
            continue;
        ++opening;
        bool blocked = false;
        for (const auto& other : c.detections) {
            if (&other == &d) continue;
            auto q = calibrationui::project(c.mapping.h, other.centre);
            if (v10Conflict(p, q, a.v10)) {
                blocked = true;
                clearance = name + " is too close to " + calibrationui::COLOUR_NAMES[other.colour] + " (X" +
                    calibrationui::fmt(abs(p.x - q.x), 0) + " Y" + calibrationui::fmt(abs(p.y - q.y), 0) +
                    " mm)";
                break;
            }
        }
        if (blocked) continue;
        double gap = norm(p - near);
        if (gap < bestGap) {
            bestGap = gap;
            best.ok = true;
            best.xy = p;
        }
    }
    best.seen = seen;
    if (!best.ok)
        best.why = !seen ? "no " + name + " block visible outside the stack"
        : !inside ? name + " block is outside the pickup workspace"
        : !unused ? name + " block already used (select ALLOW REPEATS)"
        : !aligned ? name + " block is rotated or has two possible top faces (side visible)"
        : !opening ? "gripper opening cannot clear the " + name + " face"
        : clearance;
    return best;
}
// Five consistent readings over at least 0.15 s, median, 0.5 mm packet resolution.
bool v13Sample(V13State& r, int key, Point2d p, chrono::steady_clock::time_point now, Point2d& out) {
    auto median = [&](size_t from) {
        vector<double> xs, ys;
        for (size_t i = from; i < r.samples.size(); ++i) {
            xs.push_back(r.samples[i].x);
            ys.push_back(r.samples[i].y);
        }
        sort(xs.begin(), xs.end());
        sort(ys.begin(), ys.end());
        return Point2d(xs[xs.size() / 2], ys[ys.size() / 2]);
        };
    if (key != r.sampleKey) {
        r.samples.clear();
        r.sampleTimes.clear();
        r.sampleKey = key;
    }
    if (!r.samples.empty() && norm(p - median(0)) > V13_SAME_PLACE_MM) {
        r.samples.clear();
        r.sampleTimes.clear();
    }
    if (r.sampleTimes.empty() || v13Seconds(now, r.sampleTimes.back()) >= .03) {
        r.samples.push_back(p);
        r.sampleTimes.push_back(now);
    }
    while (r.samples.size() > 7) {
        r.samples.pop_front();
        r.sampleTimes.pop_front();
    }
    if (r.samples.size() < 5 || v13Seconds(r.sampleTimes.back(), r.sampleTimes[r.sampleTimes.size() - 5]) < .15)
        return false;
    auto m = median(r.samples.size() - 5);
    out = { cvRound(m.x * 2) / 2.0, cvRound(m.y * 2) / 2.0 };
    return true;
}
// A block resting on another block: something present under this position at release has gone.
// The camera cannot measure height and the MOD picks at table height, so this is never picked.
bool v13OnAnotherBlock(const App& a, Point2d p, Point2d ignoreNear) {
    const auto& r = a.v10.v13;
    auto scene = v13Scene(a, true);
    for (const auto& old : r.snapshot) {
        if (norm(old.xy - p) > V13_UNDER_MM || norm(old.xy - ignoreNear) < V13_NEW_BLOCK_MM) continue;
        bool present = false;
        for (const auto& item : scene)
            if (item.colour == old.colour && norm(item.xy - old.xy) < 10) present = true;
        if (!present) return true;
    }
    return false;
}
// Raw sensor noise always changes some pixels. A bit-identical sparse sample for 1.5 s means the
// driver is repeating an old image: no permit or re-pick may be based on it.
void v13UpdateFrozen(V13State& r, const Mat& frame, chrono::steady_clock::time_point now) {
    if (frame.empty() || frame.type() != CV_8UC3) {
        r.frameSig.release();
        r.sameSince = {};
        r.frozen = false;
        return;
    }
    Mat sig(frame.rows / 8, frame.cols / 8, CV_8UC3);
    for (int y = 0; y < sig.rows; ++y)
        for (int x = 0; x < sig.cols; ++x) sig.at<Vec3b>(y, x) = frame.at<Vec3b>(y * 8 + 3, x * 8 + 3);
    bool same = !r.frameSig.empty() && r.frameSig.size() == sig.size() && norm(r.frameSig, sig, NORM_INF) == 0;
    if (same) {
        if (!r.sameSince.time_since_epoch().count()) r.sameSince = now;
    }
    else
        r.sameSince = {};
    r.frameSig = sig;
    r.frozen = same && v13Seconds(now, r.sameSince) >= V13_FROZEN_S;
}
// Serious, physically inconsistent states only. The robot is parked at HOME when this runs.
void v13Serious(App& a, const string& why) {
    v11OperatorRequest(a, 1);
    a.message = a.dynamicStatus = a.v10.status =
        "SERIOUS - HOLD sent: " + why + ". Robot stays at HOME. Fix the blocks by hand, then RECOVERY HELP.";
}
bool v13DropFree(const App& a, int index, Point2d p) {
    if (!v10ReturnAreaOK(a, p) || !v10ReturnVisible(a, p) || !v10Empty(a, p)) return false;
    const auto& c = *a.calibration;
    for (const auto& d : c.detections)
        if (v10Conflict(p, calibrationui::project(c.mapping.h, d.centre), a.v10)) return false;
    for (int j = 0; j < 3; ++j)
        if (j != index && abs(p.x - a.v10.drop[j].x) < a.v10.gapX + 10 && abs(p.y - a.v10.drop[j].y) < a.v10.gapY + 10)
            return false;
    return true;
}
bool v13PlanDrop(const App& a, int index, Point2d& out) {
    auto bounds = v10ActiveBounds(a);
    double best = 1e9;
    bool found = false;
    for (double x = ceil(bounds[0] / 5) * 5; x <= bounds[1]; x += 5)
        for (double y = ceil(bounds[2] / 5) * 5; y <= bounds[3]; y += 5) {
            Point2d p(x, y);
            if (!v13DropFree(a, index, p)) continue;
            double gap = norm(p - a.v10.drop[index]);
            if (gap < best) {
                best = gap;
                out = p;
                found = true;
            }
        }
    return found;
}
// A sent but not yet executed frame is resent identically (the MOD ignores a repeated attempt).
// Returns true while that wait owns the checkpoint.
bool v13ServicePending(App& a, int step, chrono::steady_clock::time_point now) {
    auto& r = a.v10.v13;
    if (!r.pendingAttempt || r.pendingStep != step) return false;
    if (r.armSeen) {
        r.pendingAttempt = 0;
        return false;
    } // robot moving: judge again once settled
    if (r.pendingTx >= V13_MAX_RESENDS) {
        r.pendingAttempt = 0;
        return false;
    } // escalate through a new attempt
    if (v13Seconds(now, r.pendingLastTx) >= V13_RESEND_S) {
        v13SendTarget(a, r.pendingMode, r.pendingStep, r.pendingAttempt, r.pendingXY, false, now);
        r.pendingLastTx = max(now, chrono::steady_clock::now());
        ++r.pendingTx;
    }
    r.hint = "RE-PICK L" + to_string(step <= 3 ? step : 7 - step) + " sent (attempt " + to_string(r.tries[step]) + "/" +
        to_string(V13_MAX_TRIES) + "), waiting for robot motion; identical resend " + to_string(r.pendingTx) +
        "/" + to_string(V13_MAX_RESENDS);
    return true;
}
// Classify a settled scene. kind 1 = transfer did not happen, 5 = return block landed elsewhere,
// 2/3/4 = stack inconsistent (serious when it persists).
// V13.5: the arm never entered the view after a release and nothing changed. The likely cause is that
// the controller did not receive the command (one-way link), not a missed grip: a re-pick sent now would
// be ignored ("not waiting at this checkpoint"). Resend the SAME frame first. The MOD ignores it when
// it was already executed: plans outside READY, permits for another step, a repeated V13 attempt.
bool v13ResendRelease(App& a, int step, chrono::steady_clock::time_point now) {
    auto& r = a.v10.v13;
    if (r.armSeen || !r.releaseStart || r.releaseResends >= V13_MAX_RELEASE_RESENDS) return false;
    bool prior = a.dynamicSending;
    a.dynamicSending = true;
    bool ok = sendFrame(a, r.releaseStart, r.releaseFields, r.releaseEnd);
    a.dynamicSending = prior;
    if (!ok) throw runtime_error("Resend of the unexecuted command failed; controller state unknown");
    ++r.releaseResends;
    auto t = max(now, chrono::steady_clock::now());
    r.releasedAt = t;
    r.clearSince = {};
    r.stillSince = {};
    r.anchorScene.clear();
    r.mismatch = 0;
    r.evidenceKind = 0;
    r.evidenceSince = {};
    r.samples.clear();
    r.sampleTimes.clear();
    r.lastProgress = t;
    a.v10.visionLast = {};
    a.v10.visionPeriod = 0;
    if (r.releaseStart == 222) a.v10.lastReturnTx = t;
    string what = r.releaseStart == 220 ? "stack plan"
        : r.releaseStart == 222 ? "return plan"
        : r.releaseStart == 234 ? "permit"
        : "camera target";
    r.hint = a.v10.status =
        "CHECK " + string(step <= 3 ? "L" : "RETURN L") + to_string(step <= 3 ? step : 7 - step) +
        ": robot never moved after the " + what +
        " (arm not seen, scene unchanged). Frame probably missed by the controller: identical resend " +
        to_string(r.releaseResends) + "/" + to_string(V13_MAX_RELEASE_RESENDS);
    return true;
}
void v13Evidence(V13State& r, int kind, chrono::steady_clock::time_point now) {
    if (kind != r.evidenceKind) {
        r.evidenceKind = kind;
        r.evidenceSince = now;
    }
}
// Decides what happened to stack layer k after the robot was sent (see SECTION 10 summary).
void v13StackSupervise(App& a, chrono::steady_clock::time_point now) {
    auto& v = a.v10;
    auto& r = v.v13;
    auto& sv = a.stackVision;
    // The checkpoint is the one the robot was released towards, NOT progress+1: progress can be
    // advanced early by the carried block seen above the stack before it was released.
    int k = r.releasedStep;
    if (k < 1 || k > 3 || sv.progress < k - 1) return;
    bool settled = v13Settled(v, now);
    bool placedFace = sv.fresh(now) && sv.topCount == 1 && sv.currentFace == v.colours[k - 1];
    if (placedFace && !settled) {
        r.pendingAttempt = 0;
        r.evidenceKind = 0;
        return;
    }
    if (!placedFace && v13ServicePending(a, k, now)) return;
    if (!settled) {
        r.evidenceKind = 0;
        r.evidenceSince = {};
        return;
    }
    if (!sv.fresh(now)) return;
    double face = max(1.0, sv.faceArea);
    auto delta = v13StackDelta(a, k);
    bool deltaAdded = v13DeltaAdded(delta, v.colours[k - 1], face);
    if (placedFace || deltaAdded) {
        // Layer k is on the stack. Record it so the swatch, the permit and the return agree.
        if (deltaAdded) r.deltaOK[k] = true;
        if (sv.progress == k - 1) {
            sv.observedLayers[k - 1] = v.colours[k - 1];
            sv.progress = k;
            sv.candidate = -2;
            sv.streak = 0;
        }
        r.pendingAttempt = 0;
        r.evidenceKind = 0;
        return;
    }
    int below = k >= 2 ? v.colours[k - 2] : -1;
    int kind = 0;
    string why;
    bool faceNotPlaced = k == 1 ? sv.emptyStable : (sv.topCount == 1 && sv.currentFace == below);
    // "Nothing changed since the robot left" is the strongest miss evidence. A face reading that
    // contradicts a large stack change (perspective side faces) is not trusted.
    // An empty stack site after verified layers is always serious, checked FIRST: a re-pick would
    // otherwise place layer k in mid-air above an empty site.
    if (k >= 2 && sv.emptyStable) {
        kind = 2;
        why = "stack zone is empty after layer " + to_string(k - 1) + " was verified (stack knocked over?)";
    }
    else if (v13DeltaNothing(delta, face) || (faceNotPlaced && !v13DeltaSomething(delta, face)))
        kind = 1;
    else if (delta.valid && v13DeltaSomething(delta, face) && (delta.blobs > 1 || delta.border)) {
        kind = 3;
        why = "a separate object changed in the stack zone (block beside the stack or fallen block)";
    }
    else if (delta.valid && delta.coloured >= .35 * face && delta.share >= .6 && delta.colour >= 0 &&
        delta.colour != v.colours[k - 1]) {
        kind = 4;
        why = string("a ") + calibrationui::COLOUR_NAMES[delta.colour] + " block was added at layer " + to_string(k) +
            " instead of " + calibrationui::COLOUR_NAMES[v.colours[k - 1]];
    }
    else if (!delta.valid && sv.topCount > 1) {
        kind = 3;
        why = "more than one block top in the stack zone";
    }
    else if (!delta.valid && sv.topCount == 1) {
        kind = 4;
        why = string("unexpected ") + calibrationui::COLOUR_NAMES[sv.currentFace] + " on top of the stack at layer " +
            to_string(k);
    }
    else {
        r.evidenceKind = 0;
        r.hint = "Layer " + to_string(k) + ": stack zone changed but the new layer is not recognised yet (change " +
            to_string(delta.area) + " px, " +
            (delta.colour >= 0 ? calibrationui::COLOUR_NAMES[delta.colour] : "no colour") + " " +
            calibrationui::fmt(delta.share * 100, 0) + "%). Robot waits at HOME";
        return;
    }
    // Premature observation (e.g. carried block seen above the stack): undo it before acting.
    if (kind == 1 && sv.progress >= k) {
        for (int l = k - 1; l < 3; ++l) sv.observedLayers[l] = -1;
        sv.progress = k - 1;
        sv.candidate = -2;
        sv.streak = 0;
    }
    v13Evidence(r, kind, now);
    double held = v13Seconds(now, r.evidenceSince);
    if (kind != 1) {
        if (held >= V13_SERIOUS_S)
            v13Serious(a, "Stack disturbed: " + why);
        else
            r.hint = "Layer " + to_string(k) + ": " + why + " (" + calibrationui::fmt(held, 1) + " s of " +
            calibrationui::fmt(V13_SERIOUS_S, 0) + " s before HOLD)";
        return;
    }
    if (held < V13_EVIDENCE_S) {
        r.hint = "Layer " + to_string(k) + ": layer not on stack after the transfer; confirming";
        return;
    }
    if (v13ResendRelease(a, k, now)) return;
    int wanted = v.colours[k - 1];
    string name = calibrationui::COLOUR_NAMES[wanted];
    if (r.tries[k] >= V13_MAX_TRIES) {
        v13Serious(a, "layer " + to_string(k) + " " + name + " missed " + to_string(r.tries[k]) +
            " re-picks: check gripper, block size and calibration");
        return;
    }
    // Only re-pick the block that is ACCOUNTED FOR: still near its source (bumped) or newly appeared
    // elsewhere (dropped). Another pre-existing block must not be picked: if the missed block is
    // still in the gripper, opening above a new target would drop it onto that block.
    Point2d source = v.pick[k - 1];
    auto accounted = [&](Point2d p) {
        if (norm(p - source) <= V13_NEAR_SOURCE_MM) return true;
        for (const auto& old : r.snapshot)
            if (old.colour == wanted && norm(old.xy - p) < V13_NEW_BLOCK_MM) return false;
        return true;
        };
    auto cand = v13FindPick(a, wanted, source, accounted);
    if (!cand.ok) {
        r.samples.clear();
        r.sampleTimes.clear();
        if (!cand.seen) {
            if (held >= V13_EVIDENCE_S + V13_LOST_ACCEPT_S) {
                v13Serious(
                    a,
                    "layer " + to_string(k) + " " + name +
                    " is neither on the stack nor anywhere in view: it may still be in the gripper or off the table");
                return;
            }
            r.hint = "MISSED L" + to_string(k) + ": " + name + " not visible; searching (" +
                calibrationui::fmt(V13_EVIDENCE_S + V13_LOST_ACCEPT_S - held, 1) +
                " s before HOLD, it may be in the gripper)";
            return;
        }
        r.hint = "MISSED L" + to_string(k) + ": " + cand.why +
            ". Block is visible, robot waits at HOME; continues automatically once it is pickable.";
        return;
    }
    if (v13OnAnotherBlock(a, cand.xy, source)) {
        v13Serious(a, "layer " + to_string(k) + " " + name +
            " landed on top of another block; its height is unknown to the robot");
        return;
    }
    Point2d xy;
    if (!v13Sample(r, k * 10 + 1, cand.xy, now, xy)) {
        r.hint = "MISSED L" + to_string(k) + ": " + name + " found at X" + calibrationui::fmt(cand.xy.x, 1) + " Y" +
            calibrationui::fmt(cand.xy.y, 1) + "; confirming position " + to_string(r.samples.size()) + "/5";
        return;
    }
    r.attempt = r.attempt % 127 + 1;
    ++r.tries[k];
    ++r.repicks;
    double moved = norm(xy - v.pick[k - 1]);
    v.pick[k - 1] = xy;
    v.sourceEvidence[k - 1].clear();
    v13SendTarget(a, 1, k, r.attempt, xy, true, now);
    r.pendingStep = k;
    r.pendingMode = 1;
    r.pendingAttempt = r.attempt;
    r.pendingXY = xy;
    r.pendingTx = 1;
    r.pendingLastTx = r.releasedAt;
    r.hint = v.status = "RE-PICK L" + to_string(k) + " " + name + " (" + to_string(r.tries[k]) + "/" +
        to_string(V13_MAX_TRIES) + "): found " + calibrationui::fmt(moved, 1) +
        " mm from the old target at X" + calibrationui::fmt(xy.x, 1) + " Y" +
        calibrationui::fmt(xy.y, 1);
    try {
        v10Journal(a);
    }
    catch (...) {}
}
// Before releasing layer `step+1`: is its block still where the plan says? 0 permit, 2 retarget, -1 wait.
int v13NextPickDecision(App& a, int step, chrono::steady_clock::time_point now, Point2d& xy, bool& fresh, string& why) {
    auto& v = a.v10;
    auto& r = v.v13;
    fresh = false;
    if (r.retargetAttempt[step]) {
        xy = r.retargetXY[step];
        return 2;
    }
    // A plain permit was already sent: the robot may be moving to the planned block. Never retarget now.
    if (r.permitKind[step] == 1) return 0;
    int n = step, colour = v.colours[n];
    string name = calibrationui::COLOUR_NAMES[colour];
    auto cand = v13FindPick(a, colour, v.pick[n]);
    if (cand.ok && norm(cand.xy - v.pick[n]) <= V13_SAME_PLACE_MM) {
        r.nextSearchSince = {};
        return 0;
    }
    if (cand.ok && v13OnAnotherBlock(a, cand.xy, v.pick[n])) {
        v13Serious(a, "next " + name + " block was knocked onto another block; its height is unknown to the robot");
        return -1;
    }
    if (!cand.ok) {
        if (!r.nextSearchSince.time_since_epoch().count()) r.nextSearchSince = now;
        why = "Next " + name + " pickup changed: " + cand.why + ". Camera keeps searching; robot waits at HOME";
        return -1;
    }
    if (!v13Sample(r, 100 + step, cand.xy, now, xy)) {
        why = "Next " + name + " block moved " + calibrationui::fmt(norm(cand.xy - v.pick[n]), 1) +
            " mm; confirming its new position";
        return -1;
    }
    r.attempt = r.attempt % 127 + 1;
    r.retargetAttempt[step] = r.attempt;
    r.retargetXY[step] = xy;
    ++r.retargets;
    fresh = true;
    v.pick[n] = xy;
    v.sourceEvidence[n].clear();
    return 2;
}
// Same idea for the return: is the next drop position still free?
int v13NextDropDecision(App& a, int step, Point2d& xy, bool& fresh) {
    auto& v = a.v10;
    auto& r = v.v13;
    fresh = false;
    if (r.retargetAttempt[step]) {
        xy = r.retargetXY[step];
        return 2;
    }
    if (r.permitKind[step] == 1) return 0; // plain permit already sent: robot uses the planned drop
    int n = 6 - step - 1;
    if (v13DropFree(a, n, v.drop[n])) return 0;
    if (!v13PlanDrop(a, n, xy)) return -1;
    r.attempt = r.attempt % 127 + 1;
    r.retargetAttempt[step] = r.attempt;
    r.retargetXY[step] = xy;
    ++r.retargets;
    fresh = true;
    v.drop[n] = xy;
    v.returnEvidence[n].clear();
    v.returnConfirmed[n] = false;
    return 2;
}
bool v13ReturnDone(const App& a, int i, chrono::steady_clock::time_point now) {
    return i >= 0 && i < 3 && ((a.v10.currentReturns[i] && a.v10.returnEvidence[i].ready(now)) || a.v10.v13.lost[i]);
}
// Same as v13StackSupervise for the return: did the stack block leave and land somewhere free?
void v13ReturnSupervise(App& a, chrono::steady_clock::time_point now) {
    auto& v = a.v10;
    auto& r = v.v13;
    const auto& sv = a.stackVision;
    int s = v.handshakeStep;
    if (s < 4 || s > 6 || r.releasedStep != s) return;
    int i = 6 - s;
    if (v13ReturnDone(a, i, now)) {
        r.pendingAttempt = 0;
        r.evidenceKind = 0;
        return;
    }
    if (v13ServicePending(a, s, now)) return;
    if (!v13Settled(v, now)) {
        r.evidenceKind = 0;
        r.evidenceSince = {};
        return;
    }
    if (!sv.fresh(now)) return;
    int stillOn = v.colours[i], below = i >= 1 ? v.colours[i - 1] : -1;
    string name = calibrationui::COLOUR_NAMES[stillOn];
    int kind = 0;
    string why;
    double face = max(1.0, sv.faceArea);
    auto delta = v13StackDelta(a, s);
    bool faceStill = sv.topCount == 1 && sv.currentFace == stillOn,
        faceRemoved = i == 0 ? sv.emptyStable : (sv.topCount == 1 && sv.currentFace == below);
    if (i >= 1 && sv.emptyStable) {
        kind = 2;
        why = "stack zone empty while lower layers should remain";
    }
    else if (v13DeltaNothing(delta, face) || (faceStill && !v13DeltaSomething(delta, face)))
        kind = 1;
    else if (v13DeltaRemoved(delta, stillOn, below, face) || (faceRemoved && !delta.valid))
        kind = 5;
    else if (delta.valid && v13DeltaSomething(delta, face) && (delta.blobs > 1 || delta.border)) {
        kind = 3;
        why = "a separate object changed in the stack zone";
    }
    else if (faceRemoved && delta.valid)
        kind = 5;
    else if (!delta.valid && sv.topCount > 1) {
        kind = 3;
        why = "more than one block top in the stack zone";
    }
    else if (!delta.valid && sv.topCount == 1) {
        kind = 4;
        why = string("unexpected ") + calibrationui::COLOUR_NAMES[sv.currentFace] + " on top of the stack";
    }
    else {
        r.evidenceKind = 0;
        r.hint = "Return L" + to_string(i + 1) + ": stack change not recognised yet (change " + to_string(delta.area) +
            " px); robot waits at HOME";
        return;
    }
    v13Evidence(r, kind, now);
    double held = v13Seconds(now, r.evidenceSince);
    if (kind >= 2 && kind <= 4) {
        if (held >= V13_SERIOUS_S)
            v13Serious(a, "Stack disturbed during return: " + why);
        else
            r.hint =
            "Return L" + to_string(i + 1) + ": " + why + " (" + calibrationui::fmt(held, 1) + " s before HOLD)";
        return;
    }
    if (held < V13_EVIDENCE_S) {
        r.hint = "Return L" + to_string(i + 1) + ": checking what the camera sees";
        return;
    }
    if (kind == 1) {
        if (v13ResendRelease(a, s, now)) return;
        // The block is still on the stack: the return pick missed. Take it again.
        if (r.tries[s] >= V13_MAX_TRIES) {
            v13Serious(a, "return of " + name + " missed " + to_string(r.tries[s]) +
                " re-picks: check gripper and stack height");
            return;
        }
        Point2d xy = v.drop[i];
        if (!v13DropFree(a, i, xy) && !v13PlanDrop(a, i, xy)) {
            r.hint = "RETURN L" + to_string(i + 1) +
                " still on stack; no free visible return position. Keep the workspace clear";
            return;
        }
        r.attempt = r.attempt % 127 + 1;
        ++r.tries[s];
        ++r.repicks;
        v.drop[i] = xy;
        v.returnEvidence[i].clear();
        v.returnConfirmed[i] = false;
        v13SendTarget(a, 1, s, r.attempt, xy, true, now);
        r.pendingStep = s;
        r.pendingMode = 1;
        r.pendingAttempt = r.attempt;
        r.pendingXY = xy;
        r.pendingTx = 1;
        r.pendingLastTx = r.releasedAt;
        r.hint = v.status = "RE-PICK RETURN L" + to_string(i + 1) + " " + name + " (" + to_string(r.tries[s]) + "/" +
            to_string(V13_MAX_TRIES) + "): block still on the stack; returning to X" +
            calibrationui::fmt(xy.x, 1) + " Y" + calibrationui::fmt(xy.y, 1);
        try {
            v10Journal(a);
        }
        catch (...) {}
        return;
    }
    // kind 5: the block left the stack but is not at its commanded return position.
    // A NEW block of that colour (absent from the scene at release) is where it landed.
    const auto& c = *a.calibration;
    int found = 0;
    Point2d best;
    double bestGap = 1e9;
    for (const auto& d : c.detections) {
        if (d.colour != stillOn || v13InStack(a, d)) continue;
        auto p = calibrationui::project(c.mapping.h, d.centre);
        bool known = false;
        for (const auto& old : r.snapshot)
            if (old.colour == stillOn && norm(old.xy - p) < V13_NEW_BLOCK_MM) known = true;
        if (known) continue;
        ++found;
        double gap = norm(p - v.drop[i]);
        if (gap < bestGap) {
            bestGap = gap;
            best = p;
        }
    }
    if (found && v13OnAnotherBlock(a, best, v.drop[i] + Point2d(1e6, 1e6))) {
        v13Serious(a, "returned " + name + " landed on top of another block; its height is unknown to the robot");
        return;
    }
    if (found) {
        Point2d xy;
        if (!v13Sample(r, 200 + s, best, now, xy)) {
            r.hint = "RETURN L" + to_string(i + 1) + " " + name + " landed " + calibrationui::fmt(bestGap, 0) +
                " mm from its target; confirming";
            return;
        }
        v.drop[i] = xy;
        v.returnEvidence[i].clear();
        ++r.accepted;
        r.hint = v.status = "RETURN L" + to_string(i + 1) + " " + name + " landed away from its target; accepted at X" +
            calibrationui::fmt(xy.x, 1) + " Y" + calibrationui::fmt(xy.y, 1) +
            " (camera confirms there)";
        return;
    }
    // Not visible anywhere: it may still be in the gripper. Picking the next stack layer would
    // open the fingers above the stack and drop it there, so this is a HOLD, not a skip.
    if (held >= V13_EVIDENCE_S + V13_LOST_ACCEPT_S) {
        v13Serious(a,
            "returned " + name +
            " left the stack but is not visible anywhere: it may still be in the gripper or off the table");
        return;
    }
    r.hint = "RETURN L" + to_string(i + 1) + " " + name + " left the stack; searching the workspace (" +
        calibrationui::fmt(V13_EVIDENCE_S + V13_LOST_ACCEPT_S - held, 1) +
        " s before HOLD, it may be in the gripper)";
}
// All three layers verified in order, the last one by the stack comparison, still true now.
bool v13StackComplete(const App& a, chrono::steady_clock::time_point now) {
    const auto& v = a.v10;
    const auto& r = v.v13;
    const auto& sv = a.stackVision;
    if (!r.deltaOK[3] || sv.progress != 3 || sv.observedLayers != v.colours || !sv.fresh(now) || !sv.clearStable ||
        sv.emptyStable)
        return false;
    if (v13Seconds(now, r.stillSince) < V13_SETTLE_S) return false;
    return v13DeltaAdded(v13StackDelta(a, 3), v.colours[2], max(1.0, sv.faceArea));
}
// A written frame is not an acknowledgement. Service the initial return handoff BEFORE
// supervisor hints can return early. Repeating permit 3 and the identical return plan
// is harmless: the MOD checks token/phase and never restarts an active return.
bool v13ReturnHandoffDue(const V10State& v, chrono::steady_clock::time_point now) {
    return v.phase == 3 && v.handshakeStep == 4 && !v.returnObserved &&
        none_of(v.returnConfirmed.begin(), v.returnConfirmed.end(), [](bool x) { return x; }) &&
        v.lastReturnTx.time_since_epoch().count() && v13Seconds(now, v.lastReturnTx) > 30;
}
bool v13RetryReturnHandoff(App& a, chrono::steady_clock::time_point now) {
    auto& v = a.v10;
    if (!v13ReturnHandoffDue(v, now) || !v10RetryAreaClear(a) ||
        (!a.stackVision.settledReturnReady(now) && !v13StackComplete(a, now)))
        return false;
    bool prior = a.dynamicSending;
    a.dynamicSending = true;
    try {
        // Send without v11SendControl: a replay must not reset release/baseline tracking.
        if (!sendFrame(a, 234, v11ControlFields(v.token, 4, 3), 235))
            throw runtime_error("Return handoff approval TX failed; inspect pendant");
        v10Send(a, true);
        v.status = "RETURN HANDOFF RETRIED: same cycle and destinations; waiting for camera confirmation";
    }
    catch (const exception& e) {
        v.phase = 5;
        v.status = e.what();
    }
    a.dynamicSending = prior;
    return true;
}

// ------------------------------------------------------------
// SECTION 11  DYNAMIC MODE: MAIN STATE MACHINE
//    Called once per camera frame. Phases: 0 ready, 1 stacking, 2 stack check, 3 returning,
//    4 finishing, 5 paused / recovery.
// ------------------------------------------------------------
void v10Tick(App& a) {
    auto& v = a.v10;
    auto& c = *a.calibration;
    if (!a.dynamicOperate || a.busy) return;
    v.exposureShift = v13ExposureShift(v.emptyBackground, c.frame);
    v.v13.hint.clear();
    if (!c.freeze) v13UpdateFrozen(v.v13, c.frame, chrono::steady_clock::now());
    auto visionNow = chrono::steady_clock::now();
    if (v.visionLast.time_since_epoch().count()) {
        double seconds = chrono::duration<double>(visionNow - v.visionLast).count();
        v.visionPeriod = v.visionPeriod > 0 ? v.visionPeriod * .8 + seconds * .2 : seconds;
    }
    v.visionLast = visionNow;
    if (c.freeze) {
        v.startRequested = false;
        v.quietSince = {};
        v.planHistory.clear();
        v.planTimes.clear();
        v.stableFrames = 0;
        for (auto& evidence : v.returnEvidence) evidence.clear();
        return;
    }
    if (c.enlargedValidation) {
        v.startRequested = false;
        v.quietSince = {};
        v.status = "Exit MEASURE AREA in TEST before Dynamic operation";
        return;
    }
    if (!c.valid || c.frame.empty()) {
        v.startRequested = false;
        v.quietSince = {};
        v.planHistory.clear();
        v.planTimes.clear();
        v.stableFrames = 0;
        for (auto& evidence : v.returnEvidence) evidence.clear();
        if (v.phase)
            v.status = "Calibration/camera unavailable. No further frames; controller stop/recovery if needed.";
        return;
    }
    if (v.phase == 0) {
        if (!v.emptyBackground.empty() &&
            (v.emptyBackground.size() != c.frame.size() || norm(Mat(v.backgroundMapping) - Mat(c.mapping.h)) > 1e-8 ||
                v.backgroundRange != v10ActiveBounds(a))) {
            v.startRequested = false;
            v.emptyBackground.release();
            v.workspaceChecked = false;
            v.stableFrames = 0;
            v.planHistory.clear();
            v.status = "Mapping/resolution/area changed: recheck workspace and recapture EMPTY background";
            return;
        }
        array<Point2d, 3> p;
        array<int, 3> col;
        bool valid = v10PickPlan(a, p, col);
        auto checkedAt = chrono::steady_clock::now();
        v10UpdatePickupSamples(v, valid, p, checkedAt);
        // V13 AUTO RUN: request the next cycle as soon as every normal Start gate passes.
        if (v.autoRun && !v.startRequested &&
            (!v.completedAt.time_since_epoch().count() || v13Seconds(checkedAt, v.completedAt) >= 1.5) &&
            v10ReadyReason(a).empty())
            v10RequestStart(a);
        if (v.startRequested) {
            // Expiry is checked BEFORE readiness so an expired click can never start later.
            if (checkedAt >= v.startDeadline) {
                v.startRequested = false;
                auto why = v10ReadyReason(a);
                v.status =
                    "START NOT SENT: " + (why.empty() ? string("request expired before fresh-frame validation") : why) +
                    ". Press Start again when Ready.";
            }
            else {
                auto why = v10ReadyReason(a);
                if (why.empty()) {
                    v.startRequested = false;
                    v10Action(a, 3);
                }
                else
                    v.status = "START WAIT: " + why + ". No robot command sent; request expires in 2 seconds.";
            }
        }
        return;
    }
    auto now = chrono::steady_clock::now();
    // V13: the timeout measures time without progress, so re-picks and waits for a block do not count against it.
    auto progressAt = max(v.cycleStarted, v.v13.lastProgress);
    if (v.phase != 5 && progressAt.time_since_epoch().count() &&
        chrono::duration<double>(now - progressAt).count() > V13_NO_PROGRESS_S) {
        v.phase = 5;
        v.status = "No progress for 10 minutes. PC paused, please check the robot.";
        v10Journal(a);
        return;
    }
    if (v.phase >= 1 && v.phase <= 3 && v.v13.frozen) {
        v.status = "CAMERA IMAGE FROZEN: identical frames for " + calibrationui::fmt(V13_FROZEN_S, 1) +
            " s. No permit or re-pick until live images resume; robot waits at HOME.";
        return;
    }
    if (v.phase >= 1 && v.phase <= 3) v13Observe(a, now);
    if (v.phase == 1 || v.phase == 2) {
        v11StackHandshake(a, now);
        if (v.phase == 5) return;
        now = chrono::steady_clock::now();
        if (a.stackVision.progress < 3 || v.handshakeStep < 3 || !v.controlLast.time_since_epoch().count()) {
            int level = max(1, min(3, v.handshakeStep));
            v.status = "Checking layer " + to_string(level) + ": " + v12StackWaitReason(a, level, now) + ".";
            v13StackSupervise(a, now);
            if (v.phase == 5) return;
            if (!v.v13.hint.empty()) v.status = v.v13.hint;
            return;
        }
        v.status = "STACK CHECK: " + a.stackVision.note;
        bool empty = true;
        // V13: only a block of the SAME colour can still be the picked block.
        for (int i = 0; i < 3; ++i) {
            for (auto& d : c.detections)
                if (d.colour == v.colours[i] && !v13InStack(a, d) &&
                    norm(calibrationui::project(c.mapping.h, d.centre) - v.pick[i]) < 25)
                    empty = false;
        }
        if (!empty) {
            // The full colour sequence is verified on the stack; a same-colour block knocked onto an
            // old source cannot be one of the stacked blocks. Do not wait on it forever.
            if (a.stackVision.settledReturnReady(now) || v13StackComplete(a, now)) {
                if (!v.v13.sourceOverrideSince.time_since_epoch().count()) v.v13.sourceOverrideSince = now;
            }
            else
                v.v13.sourceOverrideSince = {};
            if (v13Seconds(now, v.v13.sourceOverrideSince) < V13_SOURCE_OVERRIDE_S) {
                v.quietSince = {};
                v.status += ". A same-colour block sits on an old pickup position; continuing in " +
                    calibrationui::fmt(V13_SOURCE_OVERRIDE_S - v13Seconds(now, v.v13.sourceOverrideSince), 1) +
                    " s if the stack stays verified";
                return;
            }
        }
        if (v.quietSince.time_since_epoch().count() == 0) {
            v.quietSince = now;
            return;
        }
        if (chrono::duration<double>(now - v.quietSince).count() < 3) return;
        v.phase = 2;
        if (!a.stackVision.settledReturnReady(now) && !v13StackComplete(a, now)) {
            v.status = "Before returning: " + a.stackVision.note +
                string(a.finalTopFallback ? ". Waiting for the top colour and empty pickups."
                    : ". Waiting to see all three colours in order.");
            return;
        }
        if (v.handshakeStep < 3 || !v.controlLast.time_since_epoch().count()) {
            v.status = "Waiting for matching final-layer checkpoint permit";
            return;
        }
        if (!v10ReturnPlan(a)) {
            v.status =
                "WAIT: no three empty, visible return positions with clearance. Keep camera area well lit and clear; edge margins are reserved.";
            return;
        }
        try {
            v.phase = 3;
            v.handshakeStep = 4;
            v.handshakeSince = now;
            v.controlLast = {};
            v10Journal(a);
            v10Send(a, true);
            v.handshakeSince = chrono::steady_clock::now();
            v.quietSince = {};
            v.status = "Return plan sent. The top block goes back first.";
        }
        catch (const exception& e) {
            v.phase = 5;
            v.status = e.what();
        }
    }
    else if (v.phase == 3) {
        v10ObserveReturns(a, now);
        if (v.handshakeStep < 4) v.handshakeStep = 4;
        int returnIndex = 6 - min(6, v.handshakeStep);
        if (v.handshakeStep < 6 && v13ReturnDone(a, returnIndex - 1, now) && !v.v13.lost[returnIndex - 1]) {
            ++v.handshakeStep;
            v.controlLast = {};
            v.handshakeSince = now;
            returnIndex = 6 - v.handshakeStep;
        }
        if (v.handshakeStep <= 6 && v13ReturnDone(a, returnIndex, now) && a.stackVision.fresh(now)) {
            if (!v.controlLast.time_since_epoch().count() ||
                chrono::duration<double>(now - v.controlLast).count() > 2.5) {
                // V13: make sure the NEXT return position is still free before releasing the robot.
                Point2d next;
                bool fresh = false;
                int mode = v.handshakeStep < 6 ? v13NextDropDecision(a, v.handshakeStep, next, fresh) : 0;
                if (mode < 0)
                    v.v13.hint =
                    "Return L" + to_string(returnIndex + 1) +
                    " confirmed; next return position is obstructed and no free visible position exists. Keep the workspace clear";
                else if (mode == 2) {
                    v13SendTarget(a, 2, v.handshakeStep, v.v13.retargetAttempt[v.handshakeStep], next, fresh, now);
                    v.v13.permitKind[v.handshakeStep] = 2;
                    if (v.phase == 5) return;
                    v.controlLast = chrono::steady_clock::now();
                    v.handshakeSince = v.controlLast;
                    now = v.controlLast;
                }
                else {
                    v11SendControl(a, 4, v.handshakeStep);
                    v.v13.permitKind[v.handshakeStep] = 1;
                    if (v.phase == 5) return;
                    v.controlLast = chrono::steady_clock::now();
                    v.handshakeSince = v.controlLast;
                    now = v.controlLast;
                }
            }
        }
        if (v13RetryReturnHandoff(a, now)) return;
        v13ReturnSupervise(a, now);
        if (v.phase == 5) return;
        if (!v.v13.hint.empty()) {
            v.status = v.v13.hint;
            return;
        }
        if (v.handshakeSince.time_since_epoch().count() &&
            chrono::duration<double>(now - v.handshakeSince).count() > 90 && !v13ReturnDone(a, returnIndex, now)) {
            v.status = "Waiting for the return of " + string(calibrationui::COLOUR_NAMES[v.colours[returnIndex]]) +
                ": " + v.returnDetail + ".";
            return;
        }
        bool seen = false;
        for (int i = 0; i < 3; ++i)
            if (v13ReturnDone(a, i, now)) seen = true;
        if (seen) v.returnObserved = true;
        int confirmed = int(count(v.returnConfirmed.begin(), v.returnConfirmed.end(), true));
        v.status = "Returned " + to_string(confirmed) + " of 3. " + v.returnDetail;
        if (confirmed == 3 && v.handshakeStep == 6 && v.controlLast.time_since_epoch().count()) {
            v.phase = 4;
            v.finishAfter = now + chrono::seconds(int(DYNAMIC_FINISH_PAUSE_S));
            v.quietSince = {};
            v.status = "All blocks returned. Waiting for a clear view.";
            v10Journal(a);
        }
        else
            v.quietSince = {};
    }
    else if (v.phase == 4) {
        v10ObserveReturns(a, now);
        v.status = "Returned " + to_string(v10ConfirmedReturns(a, now)) + " of 3. Waiting for a clear view. " +
            a.stackVision.note;
        if (v10FinishReady(a, now)) {
            v10CompleteByCamera(a);
            v10Journal(a);
        }
    }
}

// ------------------------------------------------------------
// SECTION 12  DASHBOARD DRAWING AND MOUSE INPUT
// ------------------------------------------------------------
vector<Point2d> v10ClearanceCorners(Point2d centre, const V10State& v) {
    return { {centre.x - v.gapX, centre.y - v.gapY},
            {centre.x + v.gapX, centre.y - v.gapY},
            {centre.x + v.gapX, centre.y + v.gapY},
            {centre.x - v.gapX, centre.y + v.gapY} };
}
void drawPickupClearance(Mat& image, const App& a) {
    const auto& c = *a.calibration;
    const auto& v = a.v10;
    auto inverse = c.mapping.h.inv();
    vector<Point2d> centres;
    for (const auto& d : c.detections) centres.push_back(calibrationui::project(c.mapping.h, d.centre));
    for (size_t i = 0; i < centres.size(); ++i) {
        bool blocked = false;
        Point2d target = v13Target(a, centres[i], c.detections[i].colour);
        target = { cvRound(target.x * 2) / 2.0, cvRound(target.y * 2) / 2.0 };
        for (size_t j = 0; j < centres.size(); ++j)
            if (i != j && v10Conflict(target, centres[j], v)) blocked = true;
        vector<Point> box;
        for (auto q : v10ClearanceCorners(target, v)) {
            auto px = calibrationui::project(inverse, q);
            box.emplace_back(cvRound(px.x), cvRound(px.y));
        }
        polylines(image, box, true, blocked ? Scalar(40, 80, 245) : Scalar(150, 190, 80), 1, LINE_AA);
        vector<Point> fingers;
        for (auto offset : vector<Point2d>{ {-GRIPPER_OUTER_X_MM / 2, -GRIPPER_OUTER_Y_MM / 2},
                                           {GRIPPER_OUTER_X_MM / 2, -GRIPPER_OUTER_Y_MM / 2},
                                           {GRIPPER_OUTER_X_MM / 2, GRIPPER_OUTER_Y_MM / 2},
                                           {-GRIPPER_OUTER_X_MM / 2, GRIPPER_OUTER_Y_MM / 2} }) {
            auto px = calibrationui::project(inverse, target + offset);
            fingers.emplace_back(cvRound(px.x), cvRound(px.y));
        }
        polylines(image, fingers, true, Scalar(235, 220, 30), 2, LINE_AA);

        auto cp = calibrationui::project(inverse, target);
        drawMarker(image, cp, blocked ? Scalar(40, 80, 245) : Scalar(150, 190, 80), MARKER_CROSS, 9, 1);
        for (size_t j = i + 1; j < centres.size(); ++j)
            if (v10Conflict(target, centres[j], v) ||
                v10Conflict(v13Target(a, centres[j], c.detections[j].colour), centres[i], v)) {
                auto aPx = calibrationui::project(inverse, centres[i]),
                    bPx = calibrationui::project(inverse, centres[j]);
                line(image, aPx, bPx, Scalar(40, 80, 245), 2, LINE_AA);
            }
    }
}
void drawObservedColour(Mat& ui, Rect box, int seen) {
    if (seen >= 0 && seen < 3) {
        fill(ui, box, ink(seen + 1));
        label(ui, calibrationui::COLOUR_NAMES[seen], { box.x + 5, box.y + 20 }, .28, WHITE);
    }
    else {
        rectangle(ui, box, MUTED, 1);
        label(ui, "WAIT", { box.x + 7, box.y + 20 }, .28, MUTED);
    }
}
// Draws the DYNAMIC page: camera view with overlays on the left, cycle panel on the right.
void drawDynamic(Mat& ui, App& a) {
    auto& v = a.v10;
    auto& c = *a.calibration;
    ui = Mat(900, 1280, CV_8UC3, BG);
    Mat camera = c.frame.clone();
    drawStackVision(camera, a);
    if (c.valid) {
        if (v.showClearance) drawPickupClearance(camera, a);
        for (auto& d : c.detections) {
            auto p = calibrationui::project(c.mapping.h, d.centre);
            rectangle(camera, d.box, v10Point(a, p) ? Scalar(70, 210, 90) : GOLD, 2);
            label(camera,
                v.setupOpen ? string(calibrationui::COLOUR_NAMES[d.colour]) + " X" + calibrationui::fmt(p.x, 1) +
                " Y" + calibrationui::fmt(p.y, 1)
                : string(calibrationui::COLOUR_NAMES[d.colour]) +
                ((a.stackVision.configured && (d.box & a.stackVision.stack) == d.box) ? " (stack)"
                    : v10Point(a, p) ? ""
                    : " (outside)"),
                { d.box.x, max(15, d.box.y - 5) }, .35, WHITE);
        }
        if (!v.enlargedReturns || v.setupOpen) {
            vector<Point> region;
            for (int i : {0, 2, 8, 6}) {
                auto p = calibrationui::project(c.mapping.h.inv(), c.mapping.robot[i]);
                region.push_back(Point(cvRound(p.x), cvRound(p.y)));
            }
            polylines(camera, region, true, GOLD, 2);
        }
        vector<Point> exclusion;
        for (auto xy : vector<Point2d>{ {312.65, 105.98}, {572.65, 105.98}, {572.65, 265.98}, {312.65, 265.98} }) {
            auto p = calibrationui::project(c.mapping.h.inv(), xy);
            exclusion.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        polylines(camera, exclusion, true, Scalar(40, 90, 240), 1);
    }
    if (c.enlargedApproved && c.enlargedMatches()) {
        vector<Point> hull;
        for (auto q : c.enlargedHull) {
            auto p = calibrationui::project(c.mapping.h.inv(), Point2d(q.x, q.y));
            hull.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        polylines(camera, hull, true, Scalar(230, 210, 60), 2);
    }
    Mat resized;
    resize(camera, resized, { 820, 615 });
    resized.copyTo(ui(Rect(15, 100, 820, 615)));
    panel(ui, { 850, 100, 415, 615 });
    label(ui, "DYNAMIC STACKING", { 865, 133 }, .58, WHITE, 2);
    for (int i = 0; i < 3; ++i)
        button(ui, { 865 + i * 130, 155, 120, 38 }, "S" + to_string(i + 1),
            (v.phase && v.nextPattern ? v.nextPattern : a.selected) == i + 1);
    label(ui,
        v.phase && v.nextPattern ? "RUN S" + to_string(a.selected) + ", NEXT S" + to_string(v.nextPattern) : "TARGET",
        { 865, 222 }, .34, MUTED);
    label(ui, "LEVEL", { 995, 222 }, .31, MUTED);
    label(ui, (v.phase == 3 || v.phase == 4 || (v.phase == 5 && v.handshakeStep >= 4)) ? "RETURN OK" : "STACK SEEN",
        { 1175, 222 }, .29, MUTED);
    drawStack(ui, 865, 239, a.selected);
    for (int row = 0; row < 3; ++row) {
        int level = 2 - row;
        label(ui,
            string(level == 2 ? "Top  L3"
                : level == 1 ? "Middle  L2"
                : "Bottom  L1"),
            { 995, 252 + row * 34 }, .36, WHITE);
        bool returningView = v.phase == 3 || v.phase == 4 || (v.phase == 5 && v.handshakeStep >= 4);
        int seen = returningView
            ? (v.currentReturns[level] && v.returnEvidence[level].ready(chrono::steady_clock::now())
                ? v.colours[level]
                : -1)
            : (v.phase ? a.stackVision.observedLayers[level] : -1);
        drawObservedColour(ui, { 1175, 239 + row * 34, 70, 31 }, seen);
    }
    label(ui, "Picked bottom first, top last", { 865, 350 }, .34, MUTED);
    auto wrap = [&](const string& words, Point origin, int width, double scale = 0.42) {
        istringstream input(words);
        string word, line;
        int y = origin.y;
        while (input >> word) {
            string trial = line.empty() ? word : line + " " + word;
            int base = 0;
            if (getTextSize(trial, FONT_HERSHEY_SIMPLEX, scale, 1, &base).width > width && !line.empty()) {
                label(ui, line, { origin.x, y }, scale, WHITE);
                y += 22;
                line = word;
            }
            else
                line = trial;
        }
        if (!line.empty()) label(ui, line, { origin.x, y }, scale, WHITE);
        };
    if (v.setupOpen && v.phase == 0) {
        label(ui, "WORKSPACE SETUP", { 865, 379 }, .48, GOLD, 2);
        button(ui, { 865, 397, 185, 34 }, "ORIGINAL", !v.enlargedReturns);
        button(ui, { 1060, 397, 185, 34 }, "ENLARGED", v.enlargedReturns);
        wrap("Clear-reference setup is in CALIBRATE. Remove all blocks and the arm before capturing.", { 865, 456 }, 375);
        button(ui, { 865, 522, 380, 42 }, "LIGHT / BACKGROUND");
        button(ui, { 865, 594, 380, 38 }, "BACK TO OPERATION");
        label(ui, "Saved reference is reused between cycles and restarts.", { 865, 669 }, .35, MUTED);
    }
    else {
        string why = v.phase ? "" : v10ReadyReason(a);
        string state = v.phase == 5 ? "PAUSED"
            : v.phase == 1 ? "STACKING"
            : v.phase == 2 ? "CHECKING STACK"
            : v.phase == 3 ? "RETURNING"
            : v.phase == 4 ? "FINISHING"
            : why.empty() ? "READY"
            : "WAITING";
        fill(ui, { 865, 360, 380, 98 }, Scalar(50, 53, 48));
        label(ui, state, { 879, 388 }, .49, GOLD, 2);
        wrap(v.phase ? "Cycle running. PAUSE stops the PC only, not the robot."
            : why.empty() ? "Press START CYCLE to build the stack."
            : why,
            { 879, 416 }, 350, .40);
        button(ui, { 865, 480, 185, 34 }, "UNUSED ONLY", !v.allowRepeats);
        button(ui, { 1060, 480, 185, 34 }, "ALLOW REPEATS", v.allowRepeats);
        bool prepare = v.emptyBackground.empty() || !v.workspaceChecked;
        button(ui, { 865, 550, 380, 56 },
            v.phase == 5 ? "RECOVERY HELP"
            : v.phase ? "CYCLE RUNNING"
            : prepare ? "OPEN CALIBRATION"
            : v.startRequested ? "CHECKING FRESH FRAME"
            : "START CYCLE",
            v.phase == 5 || (!v.phase && (prepare ? true : why.empty())));
        button(ui, { 865, 630, 185, 34 }, v.phase > 0 && v.phase < 5 ? "PAUSE" : "SETUP", v.phase < 5);
        button(ui, { 1060, 630, 185, 34 }, "NEW BATCH", v.phase == 0);
        label(ui,
            string(v.enlargedReturns ? "Enlarged area" : "Original area") + "  |  " +
            (a.lightProfile == 0 ? "Light on" : "Light off") + "  |  Re-picks " + to_string(v.v13.repicks) +
            "  |  Moved " + to_string(v.v13.retargets),
            { 865, 695 }, .40, MUTED);
    }
    if (v.enlargedReturns) drawEnlargedHover(ui, a, true);
    button(ui, { 240, 105, 185, 28 }, v.autoRun ? "AUTO RUN ON" : "AUTO RUN OFF", v.autoRun);
    button(ui, { 440, 105, 185, 28 }, "RESET READINGS", v.phase == 0);
    button(ui, { 640, 105, 185, 28 }, v.showClearance ? "CLEARANCE ON" : "CLEARANCE OFF", v.showClearance);
    // The colour legend is only needed while setting up, so it does not clutter normal operation.
    if (v.showClearance && v.setupOpen) {
        label(ui, "Cyan: gripper fingers   Outer box: spacing needed   Red line: too close", { 25, 141 }, .36, GOLD);
        label(ui,
            "Spacing: X >= " + calibrationui::fmt(v.gapX, 0) + " mm or Y >= " + calibrationui::fmt(v.gapY, 0) + " mm",
            { 25, 159 }, .36, GOLD);
    }
    panel(ui, { 15, 735, 1250, 95 });
    wrap(v.status, { 28, 763 }, 1210, .42);
    label(ui,
        "Cycles done: " + to_string(a.completedCycles) +
        (v.visionPeriod > 0 ? "   |   Camera " + calibrationui::fmt(1.0 / v.visionPeriod, 1) + " fps" : ""),
        { 28, 848 }, .41, MUTED);
    label(ui, "For any fault, use the STOP button on the robot controller.", { 28, 878 }, .39, MUTED);
}

bool toggleStackVerification(App& a) {
    if (a.busy || a.sentCount || a.dynamicStep || a.buildCount || a.v10.phase) {
        a.message = a.v10.status = "Change stack verification only before Start; finish/recover active session first";
        return false;
    }
    a.finalTopFallback = !a.finalTopFallback;
    a.stackVision.finalTopOnly = a.finalTopFallback;
    a.stackVision.finalFrames = 0;
    a.stackVision.finalSince = {};
    a.message = a.v10.status =
        a.finalTopFallback
        ? "FINAL TOP fallback selected: lower colours are NOT verified. Sources must disappear and workspace must clear before return."
        : "EACH LEVEL selected: all three colours must be observed in order before return.";
    return true;
}
string appExitReason(const App& a) {
    if (a.busy) return "Exit blocked while sending. Closing the PC is not a robot stop.";
    if (a.sentCount || a.dynamicStep || a.buildCount || a.v10.phase || a.commandLatched)
        return "Exit blocked: finish the active cycle or recover on the pendant. Keep any held block secured; no blind HOME/open command.";
    return "";
}
bool requestAppExit(App& a) {
    auto why = appExitReason(a);
    if (why.empty()) return true;
    a.message = a.dynamicStatus = a.v10.status = why;
    return false;
}
bool switchPage(App& a, int page) {
    if (a.busy || a.sentCount || a.dynamicStep || a.dynamicRequest || a.buildCount || a.v10.phase ||
        a.lightCaptureActive || (a.calibration && a.calibration->diagnosticRemaining)) {
        a.dynamicStatus = "Finish current robot session before changing pages";
        a.message = a.dynamicStatus;
        return false;
    }
    if (!a.calibration) return false;
    auto& c = *a.calibration;
    if (c.entering) {
        c.message = "Finish or cancel coordinate entry before changing pages";
        return false;
    }
    a.v10.startRequested = false;
    a.pendingControllerMode = page == 0 ? 1 : page == 1 ? 4 : 3;
    if (page == 0) selectVisionProfile(a, false);
    a.dynamicOperate = false;
    a.calibrationMode = page != 0;
    a.dynamicTestMode = page == 2;
    c.exit = false;
    c.freeze = false;
    c.comparisonView = false;
    c.planning = false;
    a.mapping = false;
    a.sampleMode = false;
    a.learnPick = false;
    a.learning = false;
    a.zoneSetup = 0;
    if (page != 2) a.enlargedHover = false;
    if (page == 1) {
        c.test = false;
        c.message =
            "ONE camera calibration for Static + Dynamic. Fit once; background images are separate scene checks.";
    }
    if (page == 2) {
        c.test = true;
        c.testStable = 0;
        c.selectStable = false;
        c.selectedStability.clear();
        c.selectionTracked = false;
        c.selectionMissing = false;
        c.detectionFilter.clear();
        a.dynamicStatus =
            c.valid ? "TEST: select isolated aligned block. Send HOVER first; check pendant before PICK."
            : "TEST LOCKED: open CALIBRATE and Fit + Save, or restore the matching saved calibration.";
        c.message = a.dynamicStatus;
    }
    if (page == 0) {
        resetObservations(a);
        a.planRefreshReady = false;
        a.message = "OPERATE: fixed-coordinate demonstration; camera refreshing";
    }
    return true;
}
void drawNavigation(Mat& ui, const App& a) {
    fill(ui, Rect(0, 0, 1280, 90), Scalar(30, 26, 23));
    label(ui, "HAND EYE CONSOLE", Point(20, 28), .65, WHITE, 2);
    const char* names[] = { "STATIC", "DYNAMIC TEST", "CALIBRATE", "TEST" };
    int current = a.dynamicOperate ? 1 : a.calibrationMode ? (a.dynamicTestMode ? 3 : 2) : 0;
    for (int i = 0; i < 4; ++i) button(ui, Rect(20 + i * 155, 45, 145, 34), names[i], current == i);
    if (current == 2) {
        button(ui, Rect(660, 45, 180, 34), "FIT + SAVE");
        button(ui, Rect(850, 45, 180, 34), "INVALIDATE");
        button(ui, Rect(1040, 45, 220, 34), "LIGHT / BACKGROUND", a.stackVision.configured);
        label(ui, "ONE CAMERA MAP / STATIC + DYNAMIC", Point(720, 27), .40, GOLD);
    }
    else if (current == 3) {
        button(ui, Rect(660, 45, 180, 34), a.enlargedHover ? "ENLARGED HOVER: ON" : "ENLARGED HOVER: OFF",
            a.enlargedHover);
        button(ui, Rect(850, 45, 180, 34), "RECORD ACTUAL XY");
        button(ui, Rect(1040, 45, 220, 34), "CAPTURE DETECTION");
        label(ui,
            a.calibration && a.calibration->enlargedValidation ? "MEASURE AREA / NO PC ROBOT COMMANDS"
            : a.calibration && a.calibration->valid ? "SUPERVISED TEST / buttons can command motion"
            : "TEST LOCKED / Fit + Save calibration first",
            Point(690, 27), .40, GOLD);
    }
    else {
        button(ui, Rect(660, 45, 180, 34), "CALIBRATION SETUP", a.stackVision.configured);
        button(ui, Rect(850, 45, 230, 34), a.finalTopFallback ? "CHECK TOP ONLY" : "CHECK EACH LEVEL",
            a.finalTopFallback);
        label(ui, string(a.lightProfile == 0 ? "Light on" : "Light off"), { 1090, 65 }, .30, MUTED);
    }
    fill(ui, { 650, 0, 630, 39 }, Scalar(30, 26, 23));
    button(ui, { 660, 5, 180, 30 }, "HOLD", a.v10.phase == 5);
    button(ui, { 850, 5, 180, 30 }, "REQUEST HOME");
    button(ui, { 1040, 5, 225, 30 },
        a.v10.resetAwait ? "CONFIRM PC RESET"
        : a.v10.resetArmed ? "RESET: RECOVERED?"
        : "RESET CYCLE");
}

// ---- Mouse input
void mouse(int event, int x, int y, int, void* ptr) {
    if (event != EVENT_LBUTTONDOWN) return;
    App& a = *static_cast<App*>(ptr);
    if (Rect(660, 5, 180, 30).contains(Point(x, y))) {
        v11OperatorRequest(a, 1);
        return;
    }
    if (Rect(850, 5, 180, 30).contains(Point(x, y))) {
        v11OperatorRequest(a, 2);
        return;
    }
    if (Rect(1040, 5, 225, 30).contains(Point(x, y))) {
        v11OperatorRequest(a, 3);
        return;
    }
    if (a.busy) return;
    if (lightingMouse(a, x, y)) return;
    if (a.v10.startRequested && !Rect(865, 550, 380, 56).contains(Point(x, y))) {
        a.v10.startRequested = false;
        a.v10.status = "Pending Start cancelled by another control. No robot command sent.";
    }
    // Recovery navigation is a local view change, never a controller mode request.
    // The normal switchPage guard must continue blocking all other session changes.
    if (a.v10.phase == 5 && a.calibration && Rect(175, 45, 145, 34).contains(Point(x, y)) && !a.sentCount &&
        !a.dynamicStep && !a.dynamicRequest && !a.buildCount && !a.calibration->entering &&
        !a.calibration->diagnosticRemaining) {
        a.calibrationMode = true;
        a.dynamicOperate = true;
        a.dynamicTestMode = false;
        a.pendingControllerMode = 0;
        return;
    }
    if ((!a.calibrationMode || a.dynamicOperate) && Rect(850, 45, 230, 34).contains(Point(x, y))) {
        toggleStackVerification(a);
        return;
    }
    if ((!a.calibrationMode || a.dynamicOperate) && Rect(660, 45, 180, 34).contains(Point(x, y))) {
        switchPage(a, 1);
        return;
    }
    if (y >= 45 && y < 79) {
        for (int i = 0; i < 4; ++i)
            if (Rect(20 + i * 155, 45, 145, 34).contains(Point(x, y))) {
                if (i == 1) {
                    if (a.dynamicOperate) return;
                    if (!switchPage(a, 2)) return;
                    selectVisionProfile(a, true);
                    a.pendingControllerMode = 2;
                    a.dynamicOperate = true;
                    a.dynamicTestMode = false;
                    a.calibration->message = "DYNAMIC: choose pattern and Start when Ready";
                }
                else
                    switchPage(a, i == 0 ? 0 : i - 1);
                return;
            }
        if (a.calibrationMode) {
            if (a.dynamicStep || a.dynamicOperate) return;
            if (a.dynamicTestMode && Rect(660, 45, 180, 34).contains(Point(x, y))) {
                if (a.calibration->entering) return;
                a.enlargedHover = !a.enlargedHover;
                if (a.enlargedHover) a.calibration->enlargedValidation = false;
                a.dynamicStatus =
                    a.enlargedHover
                    ? "Enlarged HOVER enabled: approach height only outside XY coverage; PICK remains locked. Inspect physical workspace first."
                    : "Normal validated hover only";
                return;
            }
            if (!a.dynamicTestMode && Rect(1040, 45, 220, 34).contains(Point(x, y))) {
                openLightingSetup(a);
                return;
            }
            const char calActions[] = { 'c', 'i', 's' }, testActions[] = { 'x', 'e', 's' };
            for (int i = 0; i < 3; ++i)
                if (Rect(660 + i * 190, 45, i == 2 ? 220 : 180, 34).contains(Point(x, y))) {
                    try {
                        a.calibration->action(a.dynamicTestMode ? testActions[i] : calActions[i]);
                        if (a.dynamicTestMode) a.dynamicStatus = a.calibration->message;
                    }
                    catch (const exception& e) {
                        a.calibration->message = e.what();
                    }
                    return;
                }
            return;
        }
    }
    if (y < 100) return; // Integrated header owns this area; hide legacy calibration hit targets too.
    if (a.dynamicTestMode && !a.dynamicStep && !a.calibration->entering) {
        auto& c = *a.calibration;
        if (Rect(865, 700, 185, 30).contains(Point(x, y))) {
            c.enlargedValidation = !c.enlargedValidation;
            if (c.enlargedValidation) a.enlargedHover = false;
            c.freeze = false;
            c.message = c.enlargedValidation
                ? "MEASURE ONLY: select block, manually align robot, RECORD ACTUAL XY. No PC motion."
                : "Measurement mode off";
            a.dynamicStatus = c.message;
            return;
        }
        if (Rect(1060, 700, 185, 30).contains(Point(x, y))) {
            try {
                std::string why;
                bool good = c.enlargedQuality(why);
                c.enlargedApproved = good;
                c.saveEnlarged();
                if (good) {
                    c.enlargedValidation = false;
                    a.v10.emptyBackground.release();
                    a.v10.workspaceChecked = false;
                }
                c.message = why;
                a.dynamicStatus = why;
            }
            catch (const exception& e) {
                c.enlargedApproved = false;
                a.dynamicStatus = e.what();
            }
            return;
        }
        if (Rect(865, 735, 380, 20).contains(Point(x, y))) {
            try {
                c.resetEnlarged();
                a.dynamicStatus = c.message;
            }
            catch (const exception& e) {
                a.dynamicStatus = e.what();
            }
            return;
        }
    }
    if (a.dynamicOperate) {
        auto& v = a.v10;
        if (Rect(440, 105, 185, 28).contains(Point(x, y))) {
            v10ResetReadings(a);
            return;
        }
        if (Rect(640, 105, 185, 28).contains(Point(x, y))) {
            v.showClearance = !v.showClearance;
            return;
        }
        if (Rect(240, 105, 185, 28).contains(Point(x, y))) {
            if (v.phase == 5) {
                v.autoRun = false;
                return;
            }
            v.autoRun = !v.autoRun;
            v.startRequested = false;
            if (!v.phase)
                v.status = v.autoRun ? "Auto run on. The next cycle starts by itself when everything is ready."
                : "Auto run off. Press START CYCLE for each cycle.";
            return;
        }
        for (int i = 0; i < 3; ++i)
            if (Rect(865 + i * 130, 155, 120, 38).contains(Point(x, y))) {
                v10SelectPattern(a, i + 1);
                return;
            }
        if (v.setupOpen && v.phase == 0) {
            if (Rect(865, 397, 185, 34).contains(Point(x, y)))
                v10SelectReturnArea(a, false);
            else if (Rect(1060, 397, 185, 34).contains(Point(x, y)))
                v10SelectReturnArea(a, true);
            else if (Rect(865, 522, 380, 42).contains(Point(x, y))) {
                if (switchPage(a, 1)) openLightingSetup(a);
            }
            else if (Rect(865, 594, 380, 38).contains(Point(x, y)))
                v.setupOpen = false;
            return;
        }
        if (v.phase >= 1 && v.phase <= 4 && Rect(865, 630, 185, 34).contains(Point(x, y))) {
            v10Action(a, 8);
            return;
        }
        if (Rect(865, 550, 380, 56).contains(Point(x, y))) {
            if (v.phase == 5)
                v10Action(a, 6);
            else if (!v.phase) {
                if (v.emptyBackground.empty() || !v.workspaceChecked)
                    switchPage(a, 1);
                else
                    v10RequestStart(a);
            }
            return;
        }
        if (!v.phase) {
            if (Rect(865, 480, 185, 34).contains(Point(x, y))) {
                v10SetRepeatMode(a, false);
            }
            else if (Rect(1060, 480, 185, 34).contains(Point(x, y))) {
                v10SetRepeatMode(a, true);
            }
            else if (Rect(865, 630, 185, 34).contains(Point(x, y)))
                v.setupOpen = true;
            else if (Rect(1060, 630, 185, 34).contains(Point(x, y)))
                v10Action(a, 5);
        }
        return;
    }
    if (a.calibrationMode && a.dynamicTestMode && y >= 775 && y < 805 && x < 990) {
        if (a.dynamicStep) return;
        const double changes[] = { -.5, .5, -.5, .5 };
        for (int i = 0; i < 4; ++i)
            if (Rect(440 + i * 82, 775, 75, 28).contains(Point(x, y))) {
                double& value = i < 2 ? a.dynamicTrim.x : a.dynamicTrim.y;
                value = max(-5.0, min(5.0, value + changes[i]));
                saveConfig(a);
                a.dynamicStatus = "XY trim changed: verify with HOVER before pickup";
                return;
            }
        if (Rect(770, 775, 88, 28).contains(Point(x, y))) {
            a.dynamicTrim = {};
            saveConfig(a);
            return;
        }
    }
    if (a.calibrationMode && a.dynamicTestMode && y >= 775 && y < 805 && x >= 865) {
        if (a.dynamicStep) return;
        auto& g = v13Allowance();
        for (int i = 0; i < 6; ++i)
            if (Rect(865 + i * 67, 775, 63, 28).contains(Point(x, y))) {
                // CL/CR change only the colour of the selected block (red, green or blue).
                int colour = -1;
                if (i >= 2) {
                    const auto& c = *a.calibration;
                    if (c.detections.empty() || c.selectionMissing || c.selected < 0 ||
                        c.selected >= int(c.detections.size())) {
                        a.dynamicStatus = "Select a block first: CL/CR change the gain of the selected block's colour";
                        return;
                    }
                    colour = c.detections[c.selected].colour;
                    if (colour < 0 || colour > 2) return;
                    auto& slot = i < 4 ? g.colourYMinus[colour] : g.colourYPlus[colour];
                    if (!isfinite(slot)) slot = i < 4 ? g.blueY : g.centreYPlus;
                }
                double& value = i < 2 ? g.yMinus : i < 4 ? g.colourYMinus[colour] : g.colourYPlus[colour];
                value = max(0.0, min(0.2, value + (i % 2 ? .005 : -.005)));
                v13SaveAllowance();
                a.dynamicStatus = (i < 2 ? string("Left allowance ")
                    : string(i < 4 ? "Centre line Y- (left) " : "Centre line Y+ (right) ") +
                    calibrationui::COLOUR_NAMES[colour] + " ") +
                    calibrationui::fmt(value, 3) + " mm per mm saved: HOVER again to check";
                return;
            }
    }
    if (a.calibrationMode && y >= 845 && y < 875) {
        if (a.dynamicTestMode) {
            if (Rect(230, 845, 220, 30).contains(Point(x, y))) {
                a.dynamicRequest = 1;
                return;
            }
            if (Rect(460, 845, 220, 30).contains(Point(x, y))) {
                a.dynamicRequest = 2;
                return;
            }
            if (Rect(690, 845, 220, 30).contains(Point(x, y))) {
                a.dynamicRequest = 3;
                return;
            }
            if (Rect(920, 845, 200, 30).contains(Point(x, y))) {
                if (a.dynamicStep == 3) {
                    a.dynamicStep = 0;
                    a.dynamicStatus = "Operator confirmed HOME/release. Single-block test finished";
                }
                return;
            }
        }
    }
    if (a.calibrationMode && a.dynamicStep) {
        a.dynamicStatus = "Session locked: use PICK, CANCEL/REPLACE or VERIFIED HOME; controller STOP for faults";
        return;
    }
    if (a.calibrationMode && a.calibration) {
        try {
            a.calibration->click(x, y);
            if (a.dynamicTestMode) a.dynamicStatus = a.calibration->message;
        }
        catch (const exception& e) {
            a.calibration->message = e.what();
        }
        return;
    }
    if (a.busy) return;
    if (Rect(895, 574, 175, 30).contains(Point(x, y))) {
        setPickupMode(a, false);
        return;
    }
    if (Rect(1078, 574, 175, 30).contains(Point(x, y))) {
        setPickupMode(a, true);
        return;
    }
    if (Rect(895, 640, 355, 58).contains(Point(x, y))) {
        string reason = startBlock(a);
        if (reason.empty())
            operatorStart(a);
        else
            a.message = reason;
        return;
    }
    if (Rect(1080, 92, 170, 34).contains(Point(x, y)) && a.sentCount == 0) {
        a.setupOpen = !a.setupOpen;
        a.mapping = false;
        a.sampleMode = false;
        a.zoneSetup = 0;
        a.learnPick = false;
        a.learning = false;
        return;
    }
    if (a.setupOpen && a.sentCount == 0) {
        const string keys = "mvrgbpzxk";
        if (Rect(25, 833, 190, 28).contains(Point(x, y))) {
            resetSettings(a, false);
            return;
        }
        if (Rect(225, 833, 200, 28).contains(Point(x, y))) {
            resetSettings(a, true);
            return;
        }
        if (Rect(1060, 833, 190, 28).contains(Point(x, y))) {
            newBatch(a);
            return;
        }
        if (Rect(435, 833, 220, 28).contains(Point(x, y))) {
            a.pendingKey = 'l';
            return;
        }
        for (int i = 0; i < 9; ++i)
            if (Rect(25 + i * 138, 795, 129, 38).contains(Point(x, y))) {
                a.pendingKey = keys[i];
                return;
            }
    }
    if (a.sentCount > 0) return;
    if (a.learnPick && a.imageRect.contains(Point(x, y))) {
        Point pixel((x - a.imageRect.x) * a.cameraSize.width / a.imageRect.width,
            (y - a.imageRect.y) * a.cameraSize.height / a.imageRect.height);
        for (const auto& d : a.lastDetections)
            if (d.box.contains(pixel) && d.measurable) {
                a.learnCentre = d.centre;
                a.learnColour = d.colour;
                a.learnPick = false;
                a.learning = true;
                a.sizeSamples.clear();
                a.message = "Measuring selected 40x40mm face; keep still for 30 frames";
                return;
            }
        a.message = "Click an isolated, fully visible square block face";
        return;
    }
    if (a.zoneSetup > 0 && a.imageRect.contains(Point(x, y))) {
        int xx = (x - a.imageRect.x) * a.cameraSize.width / a.imageRect.width;
        int yy = (y - a.imageRect.y) * a.cameraSize.height / a.imageRect.height;
        int j = a.zoneSetup - 1;
        a.stackZones[j] = Point(xx, yy);
        a.zoneDefined[j] = true;
        a.zoneSetup = 0;
        a.stackTop[j] = 0;
        a.observedHistory[j].clear();
        saveConfig(a);
        a.message = "Saved S" + to_string(j + 1) + " stack zone; monitor visible top only";
        return;
    }
    if (a.mapping && a.imageRect.contains(Point(x, y)) && a.mapped < N) {
        int xx = (x - a.imageRect.x) * a.cameraSize.width / a.imageRect.width;
        int yy = (y - a.imageRect.y) * a.cameraSize.height / a.imageRect.height;
        a.points[a.mapped++] = Point(xx, yy);
        a.message = "Mapped P" + to_string(a.mapped);
        if (a.mapped == N) {
            a.mapping = false;
            a.verifiedMap = true;
            saveConfig(a);
            a.message = "Manual P1-P9 reference verified and saved";
        }
        return;
    }
    if (a.sampleMode && a.imageRect.contains(Point(x, y)) && !a.lastHSV.empty()) {
        int xx = (x - a.imageRect.x) * a.cameraSize.width / a.imageRect.width;
        int yy = (y - a.imageRect.y) * a.cameraSize.height / a.imageRect.height;
        int radius = 5;
        Rect r(max(0, xx - radius), max(0, yy - radius), 1, 1);
        r.width = min(2 * radius + 1, a.lastHSV.cols - r.x);
        r.height = min(2 * radius + 1, a.lastHSV.rows - r.y);
        vector<int> h;
        for (int yy2 = r.y; yy2 < r.y + r.height; ++yy2)
            for (int xx2 = r.x; xx2 < r.x + r.width; ++xx2) {
                Vec3b px = a.lastHSV.at<Vec3b>(yy2, xx2);
                if (px[1] >= 55 && px[2] >= 50) h.push_back(px[0]);
            }
        if (h.size() < 10) {
            a.message = "Sample failed: click the coloured block centre";
            return;
        }
        double sx = 0, sy = 0;
        for (int hue : h) {
            double angle = hue * 2 * CV_PI / 180.;
            sx += cos(angle);
            sy += sin(angle);
        }
        double ang = atan2(sy, sx);
        if (ang < 0) ang += 2 * CV_PI;
        a.hue[a.sampleColour] = int(round(ang * 180 / (2 * CV_PI))) % 180;
        a.tolerance[a.sampleColour] = 12;
        a.sampleMode = false;
        resetObservations(a);
        saveConfig(a);
        a.message = string("Sampled ") + cname(a.sampleColour) + " hue=" + to_string(a.hue[a.sampleColour]);
        return;
    }
    if (y >= 158 && y <= 196 && x >= 887 && x <= 1255) {
        int n = (x - 887) / 124;
        if (n >= 0 && n < 3) {
            a.selected = n + 1;
            a.releaseArmed = false;
            a.message = "Selected S" + to_string(a.selected);
        }
    }
}

void drawStack(Mat& ui, int x, int y, int routine) {
    const array<array<int, 3>, 3> top{ {{{GREEN, RED, BLUE}}, {{RED, GREEN, BLUE}}, {{BLUE, GREEN, RED}}} };
    for (int i = 0; i < 3; ++i) {
        Rect r(x, y + i * 34, 115, 31);
        fill(ui, r, ink(top[routine - 1][i]));
        label(ui, cname(top[routine - 1][i]), Point(x + 13, y + i * 34 + 21), .43, Scalar(255, 255, 255), 2);
    }
}

// ------------------------------------------------------------
// SECTION 13  SELF-TESTS AND PREVIEWS
//    Run from the command line without a camera or robot. Serial frames are captured, not sent.
// ------------------------------------------------------------
int measurementSelfTest() {
    const auto previous = fileutil::currentPath();
    try {
        string sandbox = previous + "/measurement_test_output";
        fileutil::createDirectory(sandbox);
        fileutil::currentPath(sandbox);
        for (int action = 1; action <= 3; ++action)
            for (auto xy : vector<Point2d>{ {392.7, -62.8}, {442.65, -7.8}, {492.6, 47.2} }) {
                auto b = dynamicFields(17, action, xy);
                int sum = 0;
                for (int i = 0; i < 6; ++i) {
                    if (!b[i]) throw runtime_error("Zero dynamic field");
                    sum += b[i];
                }
                Point2d decoded(100 + ((b[2] - 1) * 128 + b[3] - 1) / 2.0, -400 + ((b[4] - 1) * 128 + b[5] - 1) / 2.0);
                if (b.size() != 7 || b[6] != sum % 251 + 1 || norm(decoded - xy) > .36)
                    throw runtime_error("Dynamic protocol roundtrip failed");
            }
        bool rejected = false;
        try {
            dynamicFields(1, 1, { 600, 0 });
        }
        catch (...) {
            rejected = true;
        }
        if (!rejected) throw runtime_error("Out of range dynamic accepted");
        App occluded;
        calibrationui::App occludedCamera;
        occluded.calibration = &occludedCamera;
        occludedCamera.valid = true;
        occludedCamera.opt.source = "camera";
        occludedCamera.test = true;
        occludedCamera.selectStable = false;
        occluded.dynamicStep = 1;
        occluded.dynamicToken = 42;
        occluded.dynamicXY = { 432.1, 24.9 };
        auto locked = occluded.dynamicXY;
        if (!lockedPickupReady(occluded) || occluded.dynamicXY != locked)
            throw runtime_error("Occlusion blocked or altered locked pickup");
        occluded.dynamicStep = 0;
        if (lockedPickupReady(occluded)) throw runtime_error("Pickup allowed without hover");
        occluded.dynamicStep = 1;
        occludedCamera.valid = false;
        if (lockedPickupReady(occluded)) throw runtime_error("Pickup allowed with invalid mapping");
        cout << "PASS: occluded pickup uses locked XY; missing hover/invalid mapping still blocked\n";
        App stageGuard;
        calibrationui::App testCalibration;
        stageGuard.calibration = &testCalibration;
        stageGuard.calibrationMode = true;
        stageGuard.dynamicTestMode = true;
        dynamicAction(stageGuard, 2);
        if (stageGuard.dynamicStep || stageGuard.dynamicSending)
            throw runtime_error("Invalid dynamic stage changed state");
        cout << "PASS: dynamic base128 XY encoding/checksum, bounds rejection, guarded single-block test entry\n";
        App navigation;
        calibrationui::App navCalibration;
        navigation.calibration = &navCalibration;
        if (!switchPage(navigation, 2) || !navigation.calibrationMode || navigation.calibration->valid ||
            navigation.dynamicStatus.find("TEST LOCKED") == string::npos)
            throw runtime_error("Uncalibrated Test access/lock failed");
        if (!switchPage(navigation, 1) || !navigation.calibrationMode || navigation.dynamicTestMode)
            throw runtime_error("Calibration navigation failed");
        navCalibration.valid = true;
        if (!switchPage(navigation, 2) || !navigation.dynamicTestMode) throw runtime_error("Test navigation failed");
        navigation.dynamicStep = 1;
        if (switchPage(navigation, 0)) throw runtime_error("Active dynamic session allowed page change");
        navigation.dynamicStep = 0;
        if (!switchPage(navigation, 0) || navigation.calibrationMode) throw runtime_error("Operate navigation failed");
        cout << "PASS: Operate/Calibrate/Test navigation and active-session locks\n";
        App guard;
        guard.calibrationMode = true;
        guard.serialConnected = true;
        guard.lastSent = 91;
        if (sendCommand(guard, 240) || guard.lastSent != 91) throw runtime_error("serial calibration guard failed");
        operatorStart(guard);
        operatorReturn(guard);
        operatorClear(guard);
        operatorRelease(guard);
        if (guard.sentCount || guard.lastSent != 91) throw runtime_error("operator calibration guards failed");
        if (startBlock(guard).find("Calibration") == string::npos) throw runtime_error("calibration start not blocked");
        cout << "PASS: integrated calibration blocks serial writes and all operator motion commands\n";
        App a;
        a.mapped = N;
        a.verifiedMap = true;
        a.zoneDefined.fill(true);
        a.hue[GREEN] = 57;
        a.learnedFace = Size2f(60, 62);
        a.learnedResolution = Size(850, 620);
        resetSettings(a, false);
        App loaded;
        loadConfig(loaded);
        if (loaded.mapped || loaded.verifiedMap || loaded.autoProposal || loaded.zoneDefined[0] ||
            loaded.hue[GREEN] != 57 || loaded.learnedFace.width != 60)
            throw runtime_error("mapping reset persistence failed");
        a.sentCount = 1;
        a.mapped = 9;
        resetSettings(a, true);
        if (a.mapped != 9) throw runtime_error("active cycle reset guard failed");
        a.sentCount = 0;
        resetSettings(a, true);
        App defaults;
        loadConfig(defaults);
        if (defaults.hue[GREEN] != 60 || defaults.learnedFace.width != 0 || defaults.autoProposal)
            throw runtime_error("full reset persistence failed");
        Mat frame(400, 500, CV_8UC3, Scalar(20, 20, 20));
        RotatedRect expected(Point2f(200, 180), Size2f(64, 64), 23);
        Point2f vertices[4];
        expected.points(vertices);
        vector<Point> polygon;
        for (auto v : vertices) polygon.push_back(Point(cvRound(v.x), cvRound(v.y)));
        fillConvexPoly(frame, polygon, Scalar(0, 0, 255));
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Mat overlay = frame.clone();
        vector<Detection> det;
        findColour(hsv, overlay, RED, a, det);
        if (det.size() != 1 || !det[0].measurable || abs(det[0].face.size.width - 64) > 3 ||
            abs(det[0].face.size.height - 64) > 3)
            throw runtime_error("rotated block measurement failed");
        a.cameraSize = frame.size();
        a.lastFrame = frame;
        a.learnCentre = det[0].centre;
        a.learnColour = RED;
        a.learning = true;
        for (int i = 0; i < SIZE_SAMPLES; ++i) updateBlockCapture(a, det);
        if (a.learning || a.learnedFace.width < 60 || !fileutil::exists(a.sizeCapturePath + "/samples.csv") ||
            !fileutil::exists(a.sizeCapturePath + "/camera.png"))
            throw runtime_error("capture/export failed");
        ifstream samples(a.sizeCapturePath + "/samples.csv");
        string row;
        int rows = 0;
        while (getline(samples, row)) ++rows;
        if (rows != 31) throw runtime_error("wrong sample count");
        App restored;
        loadConfig(restored);
        if (restored.learnedResolution != frame.size() || restored.learnedFace.width != a.learnedFace.width)
            throw runtime_error("measurement reload failed");
        a.learning = true;
        a.learnCentre = det[0].centre;
        a.sizeSamples.clear();
        updateBlockCapture(a, det);
        updateBlockCapture(a, {});
        if (!a.sizeSamples.empty()) throw runtime_error("lost face did not restart stable capture");
        App batch;
        batch.autoProposal = false;
        for (int i = 0; i < N; ++i) {
            batch.positions[i].uncertain = false;
            batch.positions[i].confirmed = i % 3 + 1;
        }
        Plan initial = makePlan(batch, 1);
        if (!initial.valid) throw runtime_error("initial unused plan missing");
        batch.selected = 1;
        batch.sentPlans[0] = true;
        batch.returnSent[0] = true;
        batch.sentSources[0] = initial.source;
        batch.sentCount = 1;
        batch.sendOrder[0] = 1;
        // Return the three colours to a different permutation of the vacated slots.
        batch.returnDest[0] = { {initial.source[1], initial.source[2], initial.source[0]} };
        auto destinations = batch.returnDest[0];
        operatorRelease(batch);
        for (int i = 0; i < N; ++i) {
            batch.positions[i].uncertain = false;
            batch.positions[i].confirmed = i % 3 + 1;
        }
        for (int l = 0; l < 3; ++l) batch.positions[destinations[l]].confirmed = initial.colours[l];
        Plan next = makePlan(batch, 1);
        if (!next.valid) throw runtime_error("unused blocks not available after cycle");
        for (int src : next.source)
            if (batch.processedBlocks[src]) throw runtime_error("planner reselected USED block");
        App persisted;
        loadConfig(persisted);
        for (int dest : destinations)
            if (!persisted.processedBlocks[dest]) throw runtime_error("USED history persistence failed");
        batch.processedBlocks.fill(true);
        if (makePlan(batch, 1).valid) throw runtime_error("all USED blocks still selectable");
        auto historyBeforeMode = batch.processedBlocks;
        setPickupMode(batch, true);
        if (!makePlan(batch, 1).valid || batch.processedBlocks != historyBeforeMode)
            throw runtime_error("repeat mode failed or destroyed history");
        App modeLoaded;
        loadConfig(modeLoaded);
        if (!modeLoaded.allowRepeatPickups) throw runtime_error("repeat mode did not persist");
        batch.sentCount = 1;
        setPickupMode(batch, false);
        if (!batch.allowRepeatPickups) throw runtime_error("mode changed during active cycle");
        batch.sentCount = 0;
        setPickupMode(batch, false);
        if (makePlan(batch, 1).valid || batch.processedBlocks != historyBeforeMode)
            throw runtime_error("unused mode did not restore exclusion");
        for (int pattern = 1; pattern <= 3; ++pattern) {
            App repeated;
            repeated.autoProposal = false;
            repeated.allowRepeatPickups = true;
            for (int i = 0; i < N; ++i) {
                repeated.positions[i].uncertain = false;
                repeated.positions[i].confirmed = i % 3 + 1;
            }
            for (int run = 0; run < 10; ++run) {
                Plan plan = makePlan(repeated, pattern);
                if (!staticSourcesCurrent(repeated, plan)) throw runtime_error("Repeated static plan missing");
                int src = plan.source[0];
                repeated.positions[src].uncertain = true;
                if (staticSourcesCurrent(repeated, plan)) throw runtime_error("Stale source accepted");
                repeated.positions[src].uncertain = false;
                repeated.selected = pattern;
                int idx = pattern - 1;
                repeated.sentCount = 1;
                repeated.sendOrder[0] = pattern;
                repeated.sentPlans[idx] = true;
                repeated.returnSent[idx] = true;
                repeated.sentSources[idx] = plan.source;
                auto drop = plan.source;
                rotate(drop.begin(), drop.begin() + 1, drop.end());
                repeated.returnDest[idx] = drop;
                if (staticReturnClear(repeated, idx)) throw runtime_error("Occupied return retry accepted");
                for (int d : drop) repeated.positions[d].confirmed = UNKNOWN;
                if (!staticReturnClear(repeated, idx)) throw runtime_error("Empty return retry rejected");
                operatorRelease(repeated);
                if (repeated.sentCount || repeated.sentPlans[idx] || repeated.returnSent[idx])
                    throw runtime_error("Static cycle not released");
                for (int i = 0; i < N; ++i) repeated.positions[i].uncertain = false;
                for (int l = 0; l < 3; ++l) repeated.positions[drop[l]].confirmed = plan.colours[l];
            }
        }
        cout << "PASS: 30 static release/replan sequences across S1/S2/S3, stale-source and occupied-retry rejection\n";
        cout << "PASS: repeat/unused modes, saved preference, history preserved, active-cycle mode guard\n";
        batch.sentCount = 1;
        newBatch(batch);
        if (!batch.processedBlocks[0]) throw runtime_error("new batch allowed during active cycle");
        batch.sentCount = 0;
        newBatch(batch);
        for (bool used : batch.processedBlocks)
            if (used) throw runtime_error("new batch did not clear USED history");
        cout
            << "PASS: no-repeat planner, random-return history transfer, persistence, exhausted batch, guarded NEW BATCH reset\n";
        cout
            << "PASS: rotated contour size, 30-frame capture, CSV/image/YAML export and reload, persistent map/full resets, cycle reset guard, lost-face handling\n";
        fileutil::currentPath(previous);
        return 0;
    }
    catch (const exception& e) {
        fileutil::currentPath(previous);
        cerr << e.what() << endl;
        return 1;
    }
}

// Build tag shown on the dashboard so the running version is obvious.
static const string BUILD_TAG = string("BUILD V13.14 RETURN INSET ALL LIMITS ") + __DATE__ + " " + __TIME__;
void stackVisionSelfTest() {
    StackVision v;
    v.workspace = { 10, 10, 620, 460 };
    v.stack = { 370, 40, 230, 240 };
    v.reference = Mat(480, 640, CV_8UC3, Scalar(140, 140, 140));
    v.faceArea = 1600;
    v.configured = true;
    calibrationui::ColourModel cfg;
    cfg.staticStyle = true;
    cfg.minArea = 350;
    auto clock = chrono::steady_clock::now();
    int tick = 0;
    auto feed = [&](const Mat& f, int frames) {
        for (int k = 0; k < frames; ++k) {
            clock += chrono::milliseconds(50);
            v.observe(f, calibrationui::detect(f, cfg), clock);
            ++tick;
        }
        };
    feed(v.reference, 16);
    if (!v.startReady(clock) || v.returnReady(clock)) throw runtime_error("Visual start empty/clear gate incorrect");
    v.begin({ {2, 0, 1} });
    Mat red = v.reference.clone();
    rectangle(red, { 450, 130, 40, 40 }, Scalar(0, 0, 200), FILLED);
    feed(red, 16);
    if (v.progress) throw runtime_error("Wrong first colour accepted as full stack sequence");
    Mat blue = v.reference.clone();
    rectangle(blue, { 450, 160, 40, 40 }, Scalar(200, 0, 0), FILLED);
    feed(blue, 16);
    if (v.progress != 1) throw runtime_error("Stable blue L1 not accepted");
    Mat arm = v.reference.clone();
    rectangle(arm, { 380, 50, 200, 180 }, Scalar(0, 0, 200), FILLED);
    feed(arm, 16);
    if (v.progress != 1 || v.returnReady(clock)) throw runtime_error("Red robot-arm region counted as red stack block");
    Mat obscured = red.clone();
    rectangle(obscured, { 430, 100, 80, 25 }, Scalar(0, 0, 200), FILLED);
    feed(obscured, 16);
    if (v.progress != 2 || v.returnReady(clock)) throw runtime_error("Visible red face not latched");
    feed(red, 16);
    if (v.progress != 2) throw runtime_error("Stable red L2 not accepted");
    Mat green = v.reference.clone();
    rectangle(green, { 455, 105, 40, 40 }, Scalar(0, 200, 0), FILLED);
    feed(green, 16);
    if (v.progress != 3 || !v.returnReady(clock))
        throw runtime_error("Full blue-red-green sequence/clear view not verified");
    // STACK GUARD REMOVED: arm/obstruction pixels elsewhere in the workspace no longer block the return once the sequence was observed.
    Mat moving = green.clone();
    rectangle(moving, { 80, 80, 180, 160 }, Scalar(0, 0, 200), FILLED);
    feed(moving, 16);
    if (!v.returnReady(clock)) throw runtime_error("Return wrongly blocked by workspace obstruction (guard removed)");
    feed(green, 16);
    if (!v.returnReady(clock)) throw runtime_error("Return not ready after full sequence");
    if (v.returnReady(clock + chrono::seconds(2))) throw runtime_error("Stale workspace view allowed return command");
    feed(v.reference, 16);
    if (!v.finishReady(clock) || v.returnReady(clock))
        throw runtime_error("Empty final stack/clear view completion gate incorrect");
    for (int routine = 0; routine < 3; ++routine) {
        feed(v.reference, 16);
        array<int, 3> order;
        for (int level = 0; level < 3; ++level) order[level] = calibrationui::PATTERNS[routine][2 - level];
        v.begin(order);
        for (int level = 0; level < 3; ++level) {
            Mat placed = v.reference.clone();
            Scalar colours[3] = { Scalar(0, 0, 200), Scalar(0, 200, 0), Scalar(200, 0, 0) };
            rectangle(placed, { 450 + level * 3, 160 - level * 15, 40, 40 }, colours[order[level]], FILLED);
            feed(placed, 16);
            if (v.progress != level + 1) throw runtime_error("Selected pattern progression failed");
        }
        if (!v.returnReady(clock)) throw runtime_error("Completed pattern not ready despite clear image");
    }
    // Four individually clean glimpses separated by arm occlusion must accumulate;
    // an entirely obscured level must never be invented from the later top colour.
    feed(v.reference, 16);
    v.begin({ {2, 0, 1} });
    for (int sample = 0; sample < 4; ++sample) {
        feed(blue, 1);
        feed(arm, 2);
    }
    if (v.progress != 1) throw runtime_error("Brief occlusion erased clean placement evidence");
    feed(red, 16);
    feed(green, 16);
    if (v.progress != 3 || !v.returnReady(clock))
        throw runtime_error("Occluded sequence did not complete after real clean evidence");
    v.begin({ {2, 0, 1} });
    feed(arm, 16);
    feed(red, 16);
    feed(green, 16);
    if (v.progress || v.returnReady(clock))
        throw runtime_error("Unobserved first placement inferred from final colours");
    feed(v.reference, 16);
    v.begin({ {2, 0, 1} });
    Mat nearby = blue.clone();
    rectangle(nearby, { 380, 50, 100, 35 }, Scalar(0, 0, 200), FILLED);
    feed(nearby, 16);
    if (v.progress != 1 || v.observedLayers[0] != 2 || v.returnReady(clock))
        throw runtime_error("Clear layer face not recorded independently of distant stack guard");
    feed(red, 16);
    feed(green, 16);
    if (v.observedLayers != array<int, 3>{{2, 0, 1}} || !v.returnReady(clock))
        throw runtime_error("Layer seen-colour history/order incorrect");
    feed(v.reference, 16);
    v.begin({ {2, 1, 0} });
    Mat body = red.clone();
    rectangle(body, { 430, 145, 35, 40 }, Scalar(180, 0, 0), FILLED);
    rectangle(body, { 450, 130, 40, 40 }, Scalar(0, 0, 200), FILLED);
    auto redTops = calibrationui::detect(red, cfg);
    for (int n = 0; n < 20; ++n) {
        clock += chrono::milliseconds(50);
        v.observe(body, redTops, clock);
    }
    if (v.observedLayers[2] != -1 || v.progress || v.returnReady(clock))
        throw runtime_error("Out-of-order colour filled observed column or sequence falsely inferred");
    if (!v.topClean) throw runtime_error("Connected lower stack body falsely rejected as obstruction");
    Mat darkBody = red.clone();
    rectangle(darkBody, { 430, 145, 35, 40 }, Scalar(35, 35, 35), FILLED);
    rectangle(darkBody, { 450, 130, 40, 40 }, Scalar(0, 0, 200), FILLED);
    for (int n = 0; n < 20; ++n) {
        clock += chrono::milliseconds(50);
        v.observe(darkBody, redTops, clock);
    }
    if (!v.topClean) throw runtime_error("Dark lower stack/shadow falsely rejected as obstruction");
    Mat connectedArm = darkBody.clone();
    rectangle(connectedArm, { 380, 145, 85, 60 }, Scalar(35, 35, 35), FILLED);
    for (int n = 0; n < 20; ++n) {
        clock += chrono::milliseconds(50);
        v.observe(connectedArm, redTops, clock);
    }
    if (v.topClean || v.returnReady(clock))
        throw runtime_error("Oversized connected neutral obstruction masked as stack body");
    v.begin({ {2, 0, 1} });
    feed(blue, 16);
    feed(red, 16);
    feed(green, 16);
    if (!v.returnReady(clock)) throw runtime_error("Full sequence did not allow return");
    feed(nearby, 1);
    if (!v.returnReady(clock)) throw runtime_error("Stack surroundings still gate the return (guard not removed)");
    feed(v.reference, 20);
    feed(green, 16);
    feed(red, 1);
    // Final-top fallback is opt-in; it never writes a fictitious 3/3 history.
    v.begin({ {2, 0, 1} });
    v.finalTopOnly = true;
    feed(green, 32);
    if (v.progress || !v.returnReady(clock))
        throw runtime_error("Explicit final-top fallback failed or invented hidden levels");
    feed(red, 16);
    if (v.returnReady(clock)) throw runtime_error("Fallback accepted wrong final colour");
    feed(green, 32);
    feed(arm, 1);
    if (v.returnReady(clock)) throw runtime_error("Fallback accepted obscured final top");
    v.finalTopOnly = false;
    feed(green, 32);
    if (v.returnReady(clock)) throw runtime_error("Per-level policy bypassed missing levels");
    App fixed;
    fixed.sentSources[0] = { {0, 1, 2} };
    for (int src : {0, 1, 2}) {
        fixed.positions[src].uncertain = false;
        fixed.positions[src].confirmed = RED;
    }
    if (staticSourcesGone(fixed, 0)) throw runtime_error("Static still-present sources accepted as absent");
    for (int src : {0, 1, 2}) fixed.positions[src].confirmed = UNKNOWN;
    if (!staticSourcesGone(fixed, 0)) throw runtime_error("Static empty sources not accepted");
    fixed.positions[1].uncertain = true;
    if (staticSourcesGone(fixed, 0)) throw runtime_error("Unknown source accepted as absent");
    Mat changedSize(240, 320, CV_8UC3, Scalar(140, 140, 140));
    feed(changedSize, 1);
    if (v.configured) throw runtime_error("Resolution change retained taught stack view");
    cout
        << "PASS V10: real-image ordered stack colours, wrong sequence/red-arm/occlusion rejection, clear/empty/stale/resolution gates\n";
}
int v10SelfTest() {
    try {
        {
            StackVision dwell;
            auto now = chrono::steady_clock::now();
            dwell.configured = dwell.armed = true;
            dwell.progress = 3;
            dwell.observedLayers = dwell.expected;
            dwell.finalFrames = 8;
            dwell.finalSince = now;
            dwell.lastFrame = now;
            if (dwell.settledReturnReady(now)) throw runtime_error("Return skipped final-top settling");
            dwell.lastFrame = now + chrono::seconds(2);
            if (!dwell.settledReturnReady(dwell.lastFrame)) throw runtime_error("Settled final top blocked return");
            dwell.finalFrames = 0;
            dwell.finalSince = {};
            if (dwell.settledReturnReady(dwell.lastFrame)) throw runtime_error("Lost final top permitted return");
            cout << "PASS: two-second final-top dwell and lost-view return rejection\n";
        }
        stackVisionSelfTest();
        {
            App gate;
            calibrationui::App image;
            gate.calibration = &image;
            gate.dynamicOperate = true;
            image.valid = true;
            image.frame = Mat(480, 640, CV_8UC3, Scalar(140, 140, 140));
            gate.v10.phase = 1;
            gate.v10.cycleStarted = chrono::steady_clock::now();
            gate.v10.quietSince = chrono::steady_clock::now() - chrono::seconds(4);
            gate.lastSent = 73;
            v10Tick(gate);
            if (gate.lastSent != 73 || gate.v10.phase != 1)
                throw runtime_error("Unconfigured stack view permitted Dynamic return transmission");
            gate.stackVision.configured = true;
            gate.stackVision.armed = true;
            gate.stackVision.progress = 2;
            gate.stackVision.clearStable = true;
            gate.stackVision.topClean = true;
            gate.stackVision.top = gate.stackVision.expected[2];
            gate.stackVision.streak = 10;
            gate.stackVision.lastFrame = chrono::steady_clock::now();
            v10Tick(gate);
            if (gate.lastSent != 73 || gate.v10.phase != 1)
                throw runtime_error("Incomplete stack sequence permitted return transmission");
            gate.stackVision.progress = 3;
            gate.stackVision.clearStable = false;
            gate.stackVision.lastFrame = chrono::steady_clock::now();
            v10Tick(gate);
            if (gate.lastSent != 73 || gate.v10.phase != 1)
                throw runtime_error("Busy workspace permitted return transmission");
        }
        cout
            << "PASS V10: actual Dynamic return path blocks missing teaching, incomplete sequence and busy workspace without TX\n";

        App a;
        calibrationui::App c;
        a.calibration = &c;
        c.valid = true;
        c.opt.source = "camera";
        c.mapping.h = calibrationui::H(1, 0, 300, 0, 1, -200, 0, 0, 1);
        c.frame = Mat(480, 640, CV_8UC3, Scalar(140, 140, 140));
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 3; ++col) c.mapping.robot.push_back({ 320.0 + r * 110, -160.0 + col * 140 });
        // Set polygon indices to an ordered rectangular centre coverage.
        c.mapping.robot[0] = { 320, -160 };
        c.mapping.robot[2] = { 540, -160 };
        c.mapping.robot[8] = { 540, 120 };
        c.mapping.robot[6] = { 320, 120 };
        a.v10.emptyBackground = c.frame.clone();
        a.v10.enlargedReturns = true;
        a.dynamicOperate = true;
        if (v10Point(a, { 442, 185.98 }) || v10Point(a, { 280, 0 }))
            throw runtime_error("Restricted/uncalibrated area accepted");
        array<Point2d, 3> xy{ {{380, -90}, {450, 0}, {515, 50}} };
        for (bool ret : {false, true}) {
            auto b = v10Fields(23, 3, xy, ret);
            size_t offset = ret ? 1 : 2;
            int sum = 0;
            for (size_t i = 0; i + 1 < b.size(); ++i) sum += b[i];
            if (b.back() != sum % 251 + 1 || b.size() != (ret ? 14 : 15))
                throw runtime_error("V10 plan checksum/length failed");
            for (int i = 0; i < 3; ++i) {
                size_t j = offset + i * 4;
                Point2d decoded(100 + ((b[j] - 1) * 128 + b[j + 1] - 1) / 2.0,
                    -400 + ((b[j + 2] - 1) * 128 + b[j + 3] - 1) / 2.0);
                if (norm(decoded - xy[i]) > 1e-9) throw runtime_error("V10 XY roundtrip failed");
            }
        }
        v10SelectReturnArea(a, false);
        if (v10Point(a, { 400, -100 }) || v10ReturnAreaOK(a, { 400, -100 }))
            throw runtime_error("Original area accepted enlarged-only target");
        v10SelectReturnArea(a, true);
        if (!v10Point(a, { 400, -100 }) || !v10ReturnAreaOK(a, { 400, -100 }))
            throw runtime_error("Enlarged area rejected calibrated target");
        a.v10.phase = 3;
        v10SelectReturnArea(a, false);
        if (!a.v10.enlargedReturns) throw runtime_error("Area changed during cycle");
        a.v10.phase = 0;
        a.v10.emptyBackground = c.frame.clone();
        if (!v10ReturnPlan(a))
            throw runtime_error("Empty enlarged tested workspace did not yield three return targets");
        for (int i = 0; i < 3; ++i) {
            if (!v10Point(a, a.v10.drop[i])) throw runtime_error("Random target outside allowed region");
            for (int j = 0; j < i; ++j)
                if (v10Conflict(a.v10.drop[i], a.v10.drop[j], a.v10))
                    throw runtime_error("Return targets violate clearance");
        }
        c.frame = Mat(480, 640, CV_8UC3, Scalar(0, 0, 0));
        if (v10Empty(a, { 440, 0 })) throw runtime_error("Occluded/different background accepted EMPTY");
        if (v10ReturnPlan(a)) throw runtime_error("No-space state forced unsafe fallback");
        a.v10.phase = 3;
        a.v10.stableFrames = 20;
        a.v10.planHistory.push_back(xy);
        a.v10.quietSince = chrono::steady_clock::now();
        v10CompleteByCamera(a);
        if (a.v10.phase || a.v10.stableFrames || !a.v10.planHistory.empty() || a.completedCycles != 1 ||
            a.v10.used.size() != 3)
            throw runtime_error("Automatic camera completion/history failed");
        a.dynamicOperate = true;
        a.v10.phase = 3;
        a.v10.quietSince = chrono::steady_clock::now();
        c.freeze = true;
        v10Tick(a);
        if (a.v10.phase != 3 || a.v10.quietSince.time_since_epoch().count() != 0)
            throw runtime_error("Frozen image advanced completion");
        c.freeze = false;
        c.enlargedResolution = c.frame.size();
        c.enlargedMapping = c.mapping.h;
        c.mapping.robot = calibrationui::nominalRobot();
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 3; ++col) {
                Point2d p(300 + r * 130, -180 + col * 150);
                c.enlargedPredicted.push_back(p);
                c.enlargedActual.push_back(p + Point2d(.3, -.4));
            }
        std::string reason;
        if (!c.enlargedQuality(reason)) throw runtime_error("Measured enlarged checks rejected: " + reason);
        c.enlargedApproved = true;
        if (!c.enlargedCoverage({ 400, -100 })) throw runtime_error("Measured envelope not enabled");
        c.enlargedActual[0].x += 5;
        if (c.enlargedQuality(reason)) throw runtime_error("Bad measured XY error accepted");
        c.enlargedActual[0].x -= 5;
        c.mapping.h(0, 2) += 1;
        if (c.enlargedCoverage({ 400, -100 })) throw runtime_error("Changed mapping retained enlarged coverage");
        c.mapping.h(0, 2) -= 1;
        c.enlargedValidation = true;
        a.dynamicTestMode = true;
        a.calibrationMode = true;
        a.serialConnected = true;
        dynamicAction(a, 1);
        if (a.dynamicStatus.find("commands disabled") == string::npos || a.dynamicStep)
            throw runtime_error("Measurement mode allowed robot command");
        c.enlargedValidation = false;
        c.enlargedApproved = false;
        a.dynamicOperate = false;
        a.enlargedHover = false;
        Point2d outside(400, -100);
        if (v10HoverPoint(a, outside)) throw runtime_error("Unvalidated normal hover allowed");
        a.enlargedHover = true;
        if (!v10HoverPoint(a, outside) || v10HoverPoint(a, { 280, 0 }) || v10HoverPoint(a, { 442, 185.98 }))
            throw runtime_error("Enlarged hover bounds/exclusions wrong");
        a.dynamicStep = 1;
        a.dynamicToken = 23;
        a.dynamicXY = outside;
        c.test = true;
        if (!supervisedPickReady(a)) throw runtime_error("Operator-confirmed enlarged Test pick incorrectly blocked");
        a.dynamicStep = 2;
        if (!supervisedPickReady(a)) throw runtime_error("Same locked Pick resend blocked");
        a.lastTestTx = chrono::steady_clock::now();
        if (testCommandWait(a) <= 0) throw runtime_error("Busy-controller follow-up guard missing");
        a.lastTestTx -= chrono::seconds(30);
        if (testCommandWait(a) != 0) throw runtime_error("Follow-up wait never releases");
        a.dynamicXY = { 550, -276 };
        if (supervisedPickReady(a)) throw runtime_error("Test pick bypassed numerical limits");
        a.dynamicXY = outside;
        a.dynamicStep = 1;
        calibrationui::DetectionFilter filter;
        calibrationui::Detection d;
        d.colour = 0;
        d.centre = { 100, 100 };
        for (int i = 0; i < 8; ++i) {
            d.centre = { 100 + (i % 2 ? .8 : -.8), 100 };
            auto out = filter.update({ d });
            if (out.size() != 1 || norm(out[0].centre - Point2d(100, 100)) > 1)
                throw runtime_error("Stationary filter jitter failed");
        }
        if (!filter.update({}).empty()) throw runtime_error("Filter invented missing detection");
        d.centre = { 108, 100 };
        auto moved = filter.update({ d });
        if (norm(moved[0].centre - d.centre) > 1e-9) throw runtime_error("Filter concealed block movement");
        d.colour = 1;
        d.centre = { 108, 101 };
        if (filter.update({ d })[0].colour != 1) throw runtime_error("Track crossed colours");
        Mat diagnosticImage(180, 220, CV_8UC3, Scalar(130, 130, 130));
        rectangle(diagnosticImage, { 30, 30, 40, 40 }, Scalar(0, 0, 180), FILLED);
        circle(diagnosticImage, { 140, 100 }, 25, Scalar(0, 180, 0), FILLED);
        calibrationui::DetectionDiagnostics diag;
        auto ordinary = calibrationui::detect(diagnosticImage, c.cfg);
        auto instrumented = calibrationui::detect(diagnosticImage, c.cfg, &diag);
        if (ordinary.size() != instrumented.size() || diag.thresholdMasks[0].empty() || diag.cleanMasks[0].empty() ||
            diag.rows.size() < 2)
            throw runtime_error("Detector diagnostics changed output or missing evidence");
        bool rejected = false;
        for (auto row : diag.rows)
            if (row.find("not_quadrilateral") != string::npos) rejected = true;
        if (!rejected) throw runtime_error("Diagnostic rejection reason missing");
        Mat badShapes(200, 300, CV_8UC3, Scalar(130, 130, 130));
        rectangle(badShapes, { 20, 20, 50, 15 }, Scalar(0, 0, 180), FILLED);
        rectangle(badShapes, { 20, 20, 15, 50 }, Scalar(0, 0, 180), FILLED);
        rectangle(badShapes, { 130, 30, 28, 28 }, Scalar(0, 180, 0), FILLED);
        rectangle(badShapes, { 177, 30, 28, 28 }, Scalar(0, 180, 0), FILLED);
        rectangle(badShapes, { 157, 41, 21, 6 }, Scalar(0, 180, 0), FILLED);
        if (!calibrationui::detect(badShapes, c.cfg).empty())
            throw runtime_error("L-shape or bridged pair accepted as block");
        Mat lightingScene(240, 320, CV_8UC3, Scalar(150, 150, 150));
        rectangle(lightingScene, { 30, 90, 40, 40 }, Scalar(30, 40, 190), FILLED);
        rectangle(lightingScene, { 130, 90, 40, 40 }, Scalar(30, 170, 40), FILLED);
        rectangle(lightingScene, { 230, 90, 40, 40 }, Scalar(180, 40, 30), FILLED);
        for (double brightness : {.25, .5, 1.0, 1.5}) {
            Mat changed;
            lightingScene.convertTo(changed, -1, brightness);
            auto d = calibrationui::detect(changed, c.cfg);
            if (d.size() != 3) throw runtime_error("Auto light failed synthetic brightness " + to_string(brightness));
            array<int, 3> found{ {0, 0, 0} };
            for (auto v : d) ++found[v.colour];
            if (found != array<int, 3>{{1, 1, 1}}) throw runtime_error("Auto light changed colour identity");
        }
        Mat black(240, 320, CV_8UC3, Scalar(0, 0, 0));
        if (!calibrationui::detect(black, c.cfg).empty())
            throw runtime_error("Auto light invented blocks in black frame");
        calibrationui::LightingInfo light;
        auto adjusted = calibrationui::lightingMonitoredHSV(lightingScene, &light);
        Mat originalHSV;
        cvtColor(lightingScene, originalHSV, COLOR_BGR2HSV);
        for (int y = 0; y < adjusted.rows; y += 7)
            for (int x = 0; x < adjusted.cols; x += 7)
                if (adjusted.at<Vec3b>(y, x) != originalHSV.at<Vec3b>(y, x))
                    throw runtime_error("Lighting monitor altered raw HSV");
        calibrationui::ColourModel shared;
        shared.staticStyle = true;
        shared.minArea = 350;
        shared.sat = 65;
        shared.val = 60;
        shared.tol = { {12, 15, 15} };
        Mat paleHSV(200, 300, CV_8UC3, Scalar(0, 0, 140));
        rectangle(paleHSV, { 90, 70, 38, 38 }, Scalar(60, 42, 70), FILLED);
        Mat paleGreen;
        cvtColor(paleHSV, paleGreen, COLOR_HSV2BGR);
        auto rescuedGreen = calibrationui::detect(paleGreen, shared);
        if (rescuedGreen.size() != 1 || rescuedGreen[0].colour != 1)
            throw runtime_error("Pale dim green rescue failed");
        Mat neutralHSV(200, 300, CV_8UC3, Scalar(0, 0, 140));
        rectangle(neutralHSV, { 90, 70, 38, 38 }, Scalar(60, 15, 70), FILLED);
        Mat neutralImage;
        cvtColor(neutralHSV, neutralImage, COLOR_HSV2BGR);
        if (!calibrationui::detect(neutralImage, shared).empty())
            throw runtime_error("Green rescue accepted neutral shadow");
        cout << "PASS V10: pale/dim green rescue and neutral-shadow rejection\n";
        auto sharedBlocks = calibrationui::detect(lightingScene, shared);
        if (sharedBlocks.size() != 3) throw runtime_error("Shared detector duplicated/lost RGB blocks");
        if (!calibrationui::detect(badShapes, shared).empty() || !calibrationui::detect(black, shared).empty())
            throw runtime_error("Shared fallback accepted invalid shapes or stale objects");
        Mat circleOnly(200, 200, CV_8UC3, Scalar(130, 130, 130));
        circle(circleOnly, { 100, 100 }, 30, Scalar(0, 0, 180), FILLED);
        if (!calibrationui::detect(circleOnly, shared).empty()) throw runtime_error("Shared fallback accepted circle");
        a.enlargedHover = false;
        if (v10HoverBlockReason(a, { 384.3, -162.8 }).find("is OFF") == string::npos)
            throw runtime_error("Hover OFF diagnostic missing");
        a.enlargedHover = true;
        if (!v10HoverPoint(a, { 384.3, -162.8 }) || !v10HoverPoint(a, { 394.2, 81.5 }) || v10HoverPoint(a, { 304, -205 }) ||
            v10HoverPoint(a, { 550, -276 }))
            throw runtime_error("Separate enlarged-hover bounds/radius incorrect");
        if (v10HoverBlockReason(a, { 394.2, a.hoverMaxY + 1 }).find("CENTRE Y") == string::npos ||
            v10HoverBlockReason(a, { 304, -205 }).find("CENTRE X") == string::npos)
            throw runtime_error("Hover centre-bound diagnostic wrong");
        a.dynamicOperate = true;
        a.v10.enlargedReturns = false;
        if (v10Point(a, { 384.3, -162.8 })) throw runtime_error("Original mode bypassed original coverage");
        a.v10.enlargedReturns = true;
        if (!v10Point(a, { 384.3, -162.8 }) || !v10ReturnAreaOK(a, { 394.2, 81.5 }) || v10ReturnAreaOK(a, { 304, -205 }) ||
            v10ReturnAreaOK(a, { 550, -276 }))
            throw runtime_error("Dynamic enlarged range inconsistent with tested hover range");
        a.dynamicOperate = false;
        calibrationui::SelectedStability selectedGate;
        for (int i = 0; i < 12; ++i) selectedGate.update({ 100 + (i % 2 ? 1.8 : -1.8), 100 }, true);
        if (!selectedGate.ready) throw runtime_error("Stationary jitter never reached selected readiness");
        selectedGate.update({}, false);
        if (selectedGate.ready) throw runtime_error("Missing current selected detection remained ready");
        selectedGate.update({ 100, 100 }, true);
        if (!selectedGate.ready) throw runtime_error("Single dropout forced unnecessary full reacquisition");
        selectedGate.update({ 120, 100 }, true);
        if (selectedGate.ready) throw runtime_error("Moved block retained selected readiness");
        selectedGate.clear();
        for (int i = 0; i < 20; ++i) {
            selectedGate.update({ 100 + i * .8, 100 }, true);
            if (selectedGate.ready) throw runtime_error("Gradual moving block became selected ready");
        }
        selectedGate.clear();
        for (int i = 0; i < 12; ++i) selectedGate.update({ 100, 100 }, true);
        for (int i = 0; i < 3; ++i) selectedGate.update({}, false);
        selectedGate.update({ 100, 100 }, true);
        if (selectedGate.ready) throw runtime_error("Prolonged loss reused old readiness");
        calibrationui::App clickCheck;
        clickCheck.test = true;
        clickCheck.frame = Mat(480, 640, CV_8UC3, Scalar(130, 130, 130));
        calibrationui::Detection selectedDetection;
        selectedDetection.colour = 0;
        selectedDetection.centre = { 100, 100 };
        clickCheck.detections = { selectedDetection };
        clickCheck.selectionTracked = true;
        clickCheck.selectionPosition = { 100, 100 };
        clickCheck.selectionColour = 0;
        for (int i = 0; i < 8; ++i) clickCheck.selectedStability.update({ 100, 100 }, true);
        clickCheck.selectStable = true;
        clickCheck.testStable = 8;
        double clickScale = min(double(calibrationui::VIDEO.width) / 640, double(calibrationui::VIDEO.height) / 480);
        int clickX =
            calibrationui::VIDEO.x + int((calibrationui::VIDEO.width - 640 * clickScale) / 2 + 100 * clickScale),
            clickY =
            calibrationui::VIDEO.y + int((calibrationui::VIDEO.height - 480 * clickScale) / 2 + 100 * clickScale);
        clickCheck.click(clickX, clickY);
        if (!clickCheck.selectStable || clickCheck.selectedStability.samples.size() != 8)
            throw runtime_error("Reselecting same stationary block restarted readiness");
        clickCheck.detections[0].centre = { 150, 100 };
        clickCheck.click(clickX + int(50 * clickScale), clickY);
        if (clickCheck.selectStable || !clickCheck.selectedStability.samples.empty())
            throw runtime_error("Changed block selection retained old readiness");
        for (auto p : array<Point2d, 3>{ {{366.5, -27.5}, {394, 81.5}, {384.5, -163}} }) {
            auto hover = dynamicFields(42, 1, p);
            for (int action : {2, 3}) {
                auto follow = dynamicFields(42, action, p);
                if (follow[0] != hover[0] || !equal(hover.begin() + 2, hover.begin() + 6, follow.begin() + 2))
                    throw runtime_error("Test follow-up changed token/wire XY");
            }
        }
        for (int phase = 1; phase <= 4; ++phase) {
            App stopped;
            stopped.v10.phase = phase;
            stopped.v10.stableFrames = 20;
            stopped.lastSent = 77;
            stopped.v10.planHistory.push_back(xy);
            v10PauseForRecovery(stopped);
            if (stopped.v10.phase != 5 || stopped.v10.stableFrames || !stopped.v10.planHistory.empty() ||
                stopped.lastSent != 77)
                throw runtime_error("PC pause did not preserve recovery lock without transmission");
            stopped.dynamicOperate = true;
            if (switchPage(stopped, 0)) throw runtime_error("Recovery lock permitted mode switch");
        }
        cout << "PASS V10: PC-pause recovery from all active phases sends no bytes and preserves session lock\n";
        double projectedBlock = BLOCK_MAX_MM * (cos(BLOCK_ALIGNMENT_LIMIT_DEG * CV_PI / 180) +
            sin(BLOCK_ALIGNMENT_LIMIT_DEG * CV_PI / 180));
        if (ceil(max(projectedBlock, (GRIPPER_OUTER_X_MM + projectedBlock) / 2) + CLEARANCE_MARGIN_MM) !=
            MEASURED_GAP_X_MM ||
            ceil(max(projectedBlock, (GRIPPER_OUTER_Y_MM + projectedBlock) / 2) + CLEARANCE_MARGIN_MM) !=
            MEASURED_GAP_Y_MM)
            throw runtime_error("Measured clearance formula/constants mismatch");
        if (projectedBlock >= GRIPPER_INNER_MM) throw runtime_error("Allowed rotated block does not fit open jaws");
        cout << "PASS V10: measured71x22.8mm fingers,49.5mm opening,37mm block at12deg => X47/Y61 limits\n";
        V10State clearance;
        auto clearanceRect = v10ClearanceCorners({ 400, -20 }, clearance);
        if (clearanceRect.size() != 4 || norm(clearanceRect[0] - Point2d(353, -81)) > .001 ||
            norm(clearanceRect[2] - Point2d(447, 41)) > .001)
            throw runtime_error("Clearance overlay does not match actual exclusion limits");
        if (!v10Conflict({ 400, -20 }, { 446, 40 }, clearance) || v10Conflict({ 400, -20 }, { 447, 40 }, clearance) ||
            v10Conflict({ 400, -20 }, { 446, 41 }, clearance) || v10Conflict({ 400, -20 }, { 450, 0 }, clearance))
            throw runtime_error("Clearance OR/boundary/overlapping-box logic incorrect");
        cout << "PASS V10: clearance overlay limits, OR-axis spacing and permitted box overlap\n";
        App timed;
        timed.allowRepeatPickups = false;
        if (!timed.v10.allowRepeats) throw runtime_error("Dynamic default repeats incorrect");
        for (int phase = 1; phase <= 4; ++phase) {
            timed.v10.phase = phase;
            timed.selected = 1;
            timed.v10.nextPattern = 0;
            int beforeTx = timed.lastSent;
            if (!v10SelectPattern(timed, 3) || timed.selected != 1 || timed.v10.nextPattern != 3 ||
                timed.lastSent != beforeTx)
                throw runtime_error("Queued pattern changed active command or failed");
            v10CompleteByCamera(timed);
            if (timed.v10.phase || timed.selected != 3 || timed.v10.nextPattern)
                throw runtime_error("Next pattern not applied on completion");
        }
        timed.v10.phase = 0;
        v10SetRepeatMode(timed, false);
        if (timed.v10.allowRepeats || timed.allowRepeatPickups)
            throw runtime_error("Dynamic repeat option altered Static setting");
        v10SetRepeatMode(timed, true);
        if (!timed.v10.allowRepeats || timed.allowRepeatPickups)
            throw runtime_error("Dynamic repeat default leaked into Static");
        timed.v10.phase = 5;
        if (v10SelectPattern(timed, 2)) throw runtime_error("Recovery allowed new pattern selection");
        cout << "PASS V10: independent default repeats, queued S1/S2/S3 and completion unlock\n";
        V10State pickupWindow;
        auto sampleTime = chrono::steady_clock::now();
        array<Point2d, 3> stablePick{ {{380, -160}, {440, -40}, {500, 60}} };
        for (int f = 0; f < 19; ++f)
            v10UpdatePickupSamples(pickupWindow, true, stablePick, sampleTime + chrono::milliseconds(f * 40));
        v10UpdatePickupSamples(pickupWindow, false, {}, sampleTime + chrono::milliseconds(760));
        if (pickupWindow.stableFrames != 19) throw runtime_error("One dropout erased19 valid pickup samples");
        auto glitch = stablePick;
        glitch[0].x += 7;
        v10UpdatePickupSamples(pickupWindow, true, glitch, sampleTime + chrono::milliseconds(800));
        if (pickupWindow.stableFrames != 19 ||
            v10PickupSamplesReady(pickupWindow, glitch, sampleTime + chrono::milliseconds(800)))
            throw runtime_error("Outlier erased progress or became ready");
        v10UpdatePickupSamples(pickupWindow, true, stablePick, sampleTime + chrono::milliseconds(840));
        if (!v10PickupSamplesReady(pickupWindow, stablePick, sampleTime + chrono::milliseconds(840)))
            throw runtime_error("Brief dropout recovery failed to finish20 samples");
        auto movedPickup = stablePick;
        movedPickup[0].x += 20;
        v10UpdatePickupSamples(pickupWindow, true, movedPickup, sampleTime + chrono::milliseconds(880));
        if (pickupWindow.stableFrames != 20 ||
            v10PickupSamplesReady(pickupWindow, movedPickup, sampleTime + chrono::milliseconds(880)))
            throw runtime_error("Unconfirmed movement erased progress or became ready");
        v10UpdatePickupSamples(pickupWindow, true, movedPickup, sampleTime + chrono::milliseconds(920));
        v10UpdatePickupSamples(pickupWindow, true, movedPickup, sampleTime + chrono::milliseconds(960));
        if (pickupWindow.stableFrames != 1) throw runtime_error("Confirmed movement did not restart readings");
        v10UpdatePickupSamples(pickupWindow, false, {}, sampleTime + chrono::seconds(2));
        if (pickupWindow.stableFrames != 1 ||
            v10PickupSamplesReady(pickupWindow, movedPickup, sampleTime + chrono::seconds(2)))
            throw runtime_error("Loss erased progress or kept readiness");
        pickupWindow = V10State{};
        for (int f = 0; f < 30; ++f) {
            auto now = sampleTime + chrono::seconds(3) + chrono::milliseconds(f * 40);
            auto jitter = stablePick;
            for (auto& p : jitter) p.x += (f % 2 ? 1.5 : -1.5);
            v10UpdatePickupSamples(pickupWindow, f % 6 != 0, jitter, now);
        }
        if (!v10PickupSamplesReady(pickupWindow, stablePick, sampleTime + chrono::milliseconds(4160)))
            throw runtime_error("Intermittent loss prevented stable pickup readiness");
        if (v10PickupSamplesReady(pickupWindow, stablePick, sampleTime + chrono::seconds(6)))
            throw runtime_error("Old samples still ready");
        cout << "PASS V10: retained readings through dropout, confirmed-movement reset and stale-start rejection\n";
        v10UpdatePickupSamples(pickupWindow, false, {}, sampleTime + chrono::seconds(30));
        if (pickupWindow.stableFrames != 0 ||
            v10PickupSamplesReady(pickupWindow, stablePick, sampleTime + chrono::seconds(30)))
            throw runtime_error("Prolonged absence retained old pickup evidence");
        v10UpdatePickupSamples(pickupWindow, true, stablePick, sampleTime + chrono::seconds(31));
        if (v10PickupSamplesReady(pickupWindow, stablePick, sampleTime + chrono::seconds(31)))
            throw runtime_error("One fresh image revived expired pickup history");
        for (int f = 1; f < 5; ++f)
            v10UpdatePickupSamples(pickupWindow, true, stablePick,
                sampleTime + chrono::seconds(31) + chrono::milliseconds(f * 40));
        if (!v10PickupSamplesReady(pickupWindow, stablePick,
            sampleTime + chrono::seconds(31) + chrono::milliseconds(160)))
            throw runtime_error("Five fresh reacquisition images did not restore Ready");
        App resetCase;
        calibrationui::App resetCamera;
        resetCase.calibration = &resetCamera;
        resetCase.v10 = pickupWindow;
        resetCase.v10.workspaceChecked = true;
        resetCase.v10.emptyBackground = Mat(10, 10, CV_8UC3, Scalar(100, 100, 100));
        resetCase.v10.used.push_back({ 400, 0 });
        resetCase.lastSent = 88;
        v10ResetReadings(resetCase);
        if (resetCase.v10.stableFrames || !resetCase.v10.planHistory.empty() || !resetCase.v10.workspaceChecked ||
            resetCase.v10.emptyBackground.empty() || resetCase.v10.used.size() != 1 || resetCase.lastSent != 88)
            throw runtime_error("Reset Readings modified robot/background/history");
        resetCase.v10.phase = 3;
        resetCase.v10.stableFrames = 17;
        v10ResetReadings(resetCase);
        if (resetCase.v10.stableFrames != 17) throw runtime_error("Reset Readings allowed during movement");
        for (double angle : {0., 5., 10., 11., 20., 35., 45.}) {
            calibrationui::Detection d;
            Point2f vertices[4];
            RotatedRect({ 100, 100 }, { 40, 40 }, float(angle)).points(vertices);
            for (auto q : vertices) d.quad.emplace_back(cvRound(q.x), cvRound(q.y));
            bool expected = angle <= 11;
            if (blockAligned(d, calibrationui::H::eye()) != expected)
                throw runtime_error("Fitted alignment gate incorrect at " + to_string(angle));
        }
        calibrationui::Detection noisyFace;
        noisyFace.quad = { {80, 80}, {120, 90}, {120, 120}, {80, 120} };
        if (!blockAligned(noisyFace, calibrationui::H::eye()))
            throw runtime_error("Single noisy corner rejected straight overall face");
        Mat distantScene(200, 300, CV_8UC3, Scalar(140, 140, 140));
        rectangle(distantScene, { 30, 40, 19, 19 }, Scalar(25, 25, 65), FILLED);
        rectangle(distantScene, { 110, 40, 19, 19 }, Scalar(25, 65, 25), FILLED);
        rectangle(distantScene, { 190, 40, 19, 19 }, Scalar(65, 25, 25), FILLED);
        auto distant = calibrationui::detect(distantScene, shared);
        if (distant.size() != 3) throw runtime_error("Dim small distant RGB faces rejected");
        Mat tiny(200, 300, CV_8UC3, Scalar(140, 140, 140));
        rectangle(tiny, { 30, 40, 8, 8 }, Scalar(0, 0, 180), FILLED);
        if (!calibrationui::detect(tiny, shared).empty()) throw runtime_error("Small colour noise accepted");
        cout << "PASS V10: fitted orientation/noisy corner tolerance, rotated rejection and small dim RGB detection\n";
        App cycle;
        calibrationui::App camera;
        cycle.calibration = &camera;
        cycle.dynamicOperate = true;
        cycle.v10.enlargedReturns = true;
        cycle.v10.allowRepeats = true;
        camera.valid = true;
        camera.opt.source = "camera";
        camera.mapping.h = calibrationui::H(1, 0, 300, 0, 1, -200, 0, 0, 1);
        camera.mapping.robot = calibrationui::nominalRobot();
        camera.frame = Mat(480, 640, CV_8UC3, Scalar(140, 140, 140));
        if (!v10ReturnVisible(cycle, { 400, -20 }) || v10ReturnVisible(cycle, { cycle.hoverMinX, -20 }) ||
            v10ReturnVisible(cycle, { 400, cycle.hoverMinY }) || v10ReturnVisible(cycle, { 340, -200 }))
            throw runtime_error("Return camera/region margin incorrect");
        camera.frame.setTo(Scalar(10, 10, 10));
        if (v10ReturnVisible(cycle, { 400, -20 })) throw runtime_error("Dark return patch accepted");
        camera.frame.setTo(Scalar(140, 140, 140));
        // 24 Oct: the margin also applies to the reach arc (and every other working-area limit).
        {
            Point2d nearArc(sqrt(535.0 * 535.0 - 100.0 * 100.0), -100), wellInside(sqrt(500.0 * 500.0 - 100.0 * 100.0), -100);
            if (!v10ReturnAreaOK(cycle, nearArc) || v10ReturnAreaInset(cycle, nearArc))
                throw runtime_error("Return 5 mm inside the 540 mm reach arc was not refused");
            if (!v10ReturnAreaInset(cycle, wellInside)) throw runtime_error("Return 40 mm inside the reach arc refused");
            cycle.v10.emptyBackground = camera.frame.clone();
            for (int run = 0; run < 25; ++run) {
                if (!v10ReturnPlan(cycle)) throw runtime_error("No return plan in an empty Enlarged area");
                for (auto q : cycle.v10.drop) {
                    double radius = hypot(q.x, q.y);
                    if (radius > 540 - V10_RETURN_INSET_MM + 1e-6 || radius < 360 + V10_RETURN_INSET_MM - 1e-6 ||
                        q.x < 200 + V10_RETURN_INSET_MM - 1e-6 || !v10ReturnAreaInset(cycle, q))
                        throw runtime_error("Planned return within 30 mm of a working-area limit: X " + to_string(q.x) +
                            " Y " + to_string(q.y) + " r " + to_string(radius));
                }
            }
            cycle.v10.emptyBackground.release();
        }
        cout << "PASS V10: return inset (30 mm from rectangle, reach arcs and exclusions; 25 random plans), clipped camera view and dark-patch rejection\n";
        array<Point2d, 3> tested{ {{366, -28}, {384, -163}, {394, 81}} };
        for (int colour = 0; colour < 3; ++colour) {
            calibrationui::Detection d;
            d.colour = colour;
            d.centre = calibrationui::project(camera.mapping.h.inv(), tested[colour]);
            for (auto e : vector<Point2d>{ {-12, -12}, {12, -12}, {12, 12}, {-12, 12} })
                d.quad.emplace_back(cvRound(d.centre.x + e.x), cvRound(d.centre.y + e.y));
            d.box = boundingRect(d.quad);
            d.area = 576;
            camera.detections.push_back(d);
        }
        cycle.v10.emptyBackground = camera.frame.clone();
        for (int pattern = 1; pattern <= 3; ++pattern) {
            cycle.selected = pattern;
            array<Point2d, 3> picks;
            array<int, 3> colours;
            if (!v10PickPlan(cycle, picks, colours))
                throw runtime_error("Enlarged RGB plan rejected outside-yellow tested sources");
            for (int level = 0; level < 3; ++level)
                if (colours[level] != calibrationui::PATTERNS[pattern - 1][2 - level] ||
                    norm(picks[level] - [&] {
                        auto q = v13Target(cycle, tested[colours[level]], colours[level]);
                        return Point2d(cvRound(q.x * 2) / 2.0, cvRound(q.y * 2) / 2.0);
                        }()) > .01)
                    throw runtime_error("Enlarged pattern bottom-first order incorrect");
            if (!v10ReturnPlan(cycle)) throw runtime_error("Enlarged return sampler could not use extended area");
            for (int i = 0; i < 3; ++i) {
                if (!enlargedHoverPoint(cycle, cycle.v10.drop[i]))
                    throw runtime_error("Return escaped enlarged receiver limits");
                for (auto occupied : tested)
                    if (v10Conflict(cycle.v10.drop[i], occupied, cycle.v10))
                        throw runtime_error("Return overlapped detected block");
                for (int j = 0; j < i; ++j)
                    if (v10Conflict(cycle.v10.drop[i], cycle.v10.drop[j], cycle.v10))
                        throw runtime_error("Enlarged returns violated gripper spacing");
            }
        }
        cycle.v10.workspaceChecked = true;
        v10SelectReturnArea(cycle, false);
        if (!cycle.v10.emptyBackground.empty() || cycle.v10.workspaceChecked)
            throw runtime_error("Area switch reused old background approval");
        cycle.v10.enlargedReturns = true;
        cycle.selected = 1;
        cycle.v10.used.clear();
        cycle.v10.allowRepeats = true;
        auto closeBlue = camera.detections[2];
        auto closeCentre = calibrationui::project(camera.mapping.h.inv(), Point2d(355, -75));
        auto shift = closeCentre - closeBlue.centre;
        closeBlue.centre = closeCentre;
        for (auto& q : closeBlue.quad) q += Point(cvRound(shift.x), cvRound(shift.y));
        closeBlue.box = boundingRect(closeBlue.quad);
        camera.detections.insert(camera.detections.begin(), closeBlue);
        auto alternativeRed = camera.detections[1];
        Point2d alternativeRedXY(470, -45);
        auto redCentre = calibrationui::project(camera.mapping.h.inv(), alternativeRedXY);
        auto redShift = redCentre - alternativeRed.centre;
        alternativeRed.centre = redCentre;
        for (auto& q : alternativeRed.quad) q += Point(cvRound(redShift.x), cvRound(redShift.y));
        alternativeRed.box = boundingRect(alternativeRed.quad);
        camera.detections.push_back(alternativeRed);
        array<Point2d, 3> alternatives;
        array<int, 3> altColours;
        string why;
        if (!v10PickPlan(cycle, alternatives, altColours, &why) || norm(alternatives[0] - tested[2]) > .01)
            throw runtime_error("Planner failed to try a valid alternative colour combination");
        cycle.v10.allowRepeats = false;
        cycle.v10.used = { tested[0], alternativeRedXY };
        if (v10PickPlan(cycle, alternatives, altColours, &why) || why.find("already used") == string::npos)
            throw runtime_error("Readiness did not explain used-colour block");
        cycle.v10.used.clear();
        cycle.v10.allowRepeats = true;
        cycle.v10.emptyBackground.release();
        cycle.v10.workspaceChecked = false;
        v10Action(cycle, 7);
        if (!cycle.v10.emptyBackground.empty() || cycle.v10.workspaceChecked)
            throw runtime_error("One-button Prepare captured occupied area");
        camera.detections.clear();
        v10Action(cycle, 7);
        if (cycle.v10.emptyBackground.empty() || !cycle.v10.workspaceChecked || cycle.v10.phase)
            throw runtime_error("One-button empty preparation failed or commanded robot");
        cycle.v10.phase = 4;
        cycle.v10.colours = { {0, 1, 2} };
        cycle.v10.drop = tested;
        auto clockNow = chrono::steady_clock::now();
        cycle.v10.finishAfter = clockNow + chrono::seconds(25);
        for (int i = 0; i < 3; ++i) {
            calibrationui::Detection d;
            d.colour = i;
            d.centre = calibrationui::project(camera.mapping.h.inv(), tested[i]);
            camera.detections.push_back(d);
        }
        // Brief asynchronous dropouts must not require all three colours in the same frame.
        for (int frame = 0; frame < 20; ++frame) {
            auto now = clockNow + chrono::milliseconds(frame * 50);
            for (int i = 0; i < 3; ++i) cycle.v10.returnEvidence[i].update((frame + i) % 5 != 0, now);
        }
        if (v10FinishReady(cycle, clockNow + chrono::seconds(1))) throw runtime_error("Finishing pause released early");
        auto finishTime = clockNow + chrono::seconds(26);
        cycle.stackVision.configured = true;
        cycle.stackVision.clearStable = true;
        cycle.stackVision.emptyStable = true;
        cycle.stackVision.lastFrame = finishTime;
        for (int f = 0; f < 4; ++f)
            for (int i = 0; i < 3; ++i) cycle.v10.returnEvidence[i].update(true, finishTime);
        if (!v10FinishReady(cycle, finishTime)) throw runtime_error("Asynchronous return evidence never completed");
        if (v10FinishReady(cycle, finishTime + chrono::seconds(1)))
            throw runtime_error("Stale return evidence released cycle");
        cycle.v10.returnEvidence[2].clear();
        for (int f = 0; f < 20; ++f) cycle.v10.returnEvidence[2].update(false, finishTime);
        if (v10FinishReady(cycle, finishTime)) throw runtime_error("Missing return accepted");
        if (v10RetryAreaClear(cycle)) throw runtime_error("Return retry ignored newly occupied destinations");
        for (int pattern = 1; pattern <= 3; ++pattern)
            for (int run = 0; run < 10; ++run) {
                cycle.selected = pattern;
                cycle.v10.phase = 4;
                cycle.v10.drop = tested;
                cycle.v10.colours = { {0, 1, 2} };
                cycle.v10.finishAfter = finishTime;
                for (auto& evidence : cycle.v10.returnEvidence) {
                    evidence.clear();
                    for (int f = 0; f < 20; ++f) evidence.update(f % 5 != 0, finishTime);
                }
                if (!v10FinishReady(cycle, finishTime))
                    throw runtime_error("Repeated Dynamic completion remained locked");
                int before = cycle.completedCycles;
                v10CompleteByCamera(cycle);
                if (cycle.v10.phase || cycle.v10.stableFrames || !cycle.v10.planHistory.empty() ||
                    cycle.completedCycles != before + 1)
                    throw runtime_error("Dynamic completion did not clear locks exactly once");
                // Recreate separated aligned returned RGB blocks to test the next pattern's eligibility.
                camera.detections.clear();
                for (int i = 0; i < 3; ++i) {
                    calibrationui::Detection d;
                    d.colour = i;
                    d.centre = calibrationui::project(camera.mapping.h.inv(), tested[i]);
                    for (auto e : vector<Point2d>{ {-12, -12}, {12, -12}, {12, 12}, {-12, 12} })
                        d.quad.emplace_back(cvRound(d.centre.x + e.x), cvRound(d.centre.y + e.y));
                    d.box = boundingRect(d.quad);
                    d.area = 576;
                    camera.detections.push_back(d);
                }
                cycle.v10.allowRepeats = true;
                if (!v10PickPlan(cycle, alternatives, altColours, &why))
                    throw runtime_error("Repeat Dynamic next selection blocked: " + why);
                cycle.v10.allowRepeats = false;
                if (v10PickPlan(cycle, alternatives, altColours, &why) || why.find("ALLOW REPEATS") == string::npos)
                    throw runtime_error("Unused-only missing repeat explanation");
            }
        cycle.v10.phase = 3;
        cycle.v10.colours = { {0, 1, 2} };
        cycle.v10.drop = tested;
        camera.detections.clear();
        for (int i = 0; i < 3; ++i) {
            calibrationui::Detection d;
            d.colour = (i + 1) % 3;
            d.centre = calibrationui::project(camera.mapping.h.inv(), tested[i]);
            camera.detections.push_back(d);
        }
        for (auto& e : cycle.v10.returnEvidence) e.clear();
        for (int f = 0; f < 20; ++f) v10ObserveReturns(cycle, finishTime);
        if (v10ConfirmedReturns(cycle, finishTime)) throw runtime_error("Wrong return colours accepted");
        for (int i = 0; i < 3; ++i) {
            camera.detections[i].colour = i;
            camera.detections[i].centre += Point2d(20, 0);
        }
        for (auto& e : cycle.v10.returnEvidence) e.clear();
        for (int f = 0; f < 20; ++f) v10ObserveReturns(cycle, finishTime);
        if (v10ConfirmedReturns(cycle, finishTime)) throw runtime_error("Displaced returns accepted");
        cycle.v10.returnConfirmed.fill(false);
        for (auto& e : cycle.v10.returnEvidence) e.clear();
        cycle.v10.colours = { {0, 1, 2} };
        cycle.v10.drop = tested;
        for (int colour = 0; colour < 3; ++colour) {
            camera.detections.clear();
            calibrationui::Detection d;
            d.colour = colour;
            d.centre = calibrationui::project(camera.mapping.h.inv(), tested[colour]);
            camera.detections.push_back(d);
            for (int f = 0; f < 12; ++f) v10ObserveReturns(cycle, finishTime + chrono::seconds(colour * 2));
        }
        if (count(cycle.v10.returnConfirmed.begin(), cycle.v10.returnConfirmed.end(), true) != 3)
            throw runtime_error("Independent returns were not latched");
        cycle.v10.phase = 4;
        cycle.v10.finishAfter = finishTime + chrono::seconds(25);
        if (v10FinishReady(cycle, finishTime + chrono::seconds(26)))
            throw runtime_error("Latched evidence bypassed final fresh-position check");
        camera.detections.clear();
        for (int colour = 0; colour < 3; ++colour) {
            calibrationui::Detection d;
            d.colour = colour;
            d.centre = calibrationui::project(camera.mapping.h.inv(), tested[colour]);
            camera.detections.push_back(d);
        }
        for (int f = 0; f < 20; ++f) v10ObserveReturns(cycle, finishTime + chrono::seconds(26));
        cycle.stackVision.lastFrame = finishTime + chrono::seconds(26);
        if (!v10FinishReady(cycle, finishTime + chrono::seconds(26)))
            throw runtime_error("Fresh final returns failed to unlock completion");
        cout
            << "PASS V10: sequential return confirmation survives occlusion but final completion requires fresh return detections\n";
        cout
            << "PASS V10: 30 Dynamic completion/replan sequences, asynchronous dropouts, missing/stale-return rejection and repeat-policy explanation\n";
        cout << "PASS V10: all-visible pickup clearance, occupied retry rejection and visual completion gate\n";
        cout << "PASS V10: alternative RGB combinations, actionable readiness, one-button empty-only preparation\n";
        cout
            << "PASS V10: all 3 Dynamic patterns use outside-yellow tested pickup XY and enlarged empty returns with unchanged spacing\n";
        cout
            << "PASS V10: stationary jitter readiness, brief dropout recovery, current loss/movement/drift rejection\n";
        cout
            << "PASS V10: shared Static segmentation with fresh dim fallback; no duplicates/circles/merged shapes/stale objects\n";
        cout
            << "PASS V10: RAW HSV lighting rollback, exact unchanged HSV, synthetic RGB brightness fixtures and black-frame rejection\n";
        cout << "PASS V10: detection masks/rejection logging preserve detector acceptance\n";
        cout << "PASS V10: fresh-only median tracking, movement reset, missing detection and colour separation\n";
        cout
            << "PASS V10: enlarged hover eligibility, workspace exclusions, supervised locked enlarged PICK eligibility\n";
        cout << "PASS V10: enlarged checks, error gate, mapping change rejection, measure-only motion guard\n";
        cout << "PASS V10: automatic completion/history and frozen-image guard\n";
        cout
            << "PASS V10: atomic coordinate plans, checksum, Original/Enlarged restriction, stack exclusion, X/Y clearance, empty-background rejection, no-space rejection\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}

int v10Preview() {
    App a;
    calibrationui::App c;
    a.calibration = &c;
    c.opt.source = "demo";
    for (int i = 0; i < 25; ++i) {
        c.frame = calibrationui::synthetic(c);
        c.update();
    }
    if (!calibrationui::makeMapping(c.grid, c.refs, c.frame.size(), c.mapping)) return 1;
    c.valid = true;
    c.test = true;
    c.frame = calibrationui::synthetic(c);
    c.update();
    a.dynamicOperate = true;
    a.calibrationMode = true;
    a.v10.status = "SYNTHETIC PREVIEW: verify workspace and capture EMPTY background before live operation";
    drawDynamic(c.ui, a);
    drawNavigation(c.ui, a);
    if (!imwrite("chloec10/dynamic_preview.png", c.ui)) return 1;
    c.frame = Mat(300, 320, CV_8UC3, Scalar(170, 170, 170));
    c.mapping.h = calibrationui::H(1, 0, 300, 0, 1, -200, 0, 0, 1);
    c.detections.clear();
    for (int i = 0; i < 3; ++i) {
        Point centre(80 + i * 45, 100 + (i % 2) * 50);
        calibrationui::Detection d;
        d.centre = centre;
        d.colour = i;
        d.box = { centre.x - 18, centre.y - 18, 36, 36 };
        rectangle(c.frame, d.box, ink(i + 1), FILLED);
        d.quad = { {centre.x - 18, centre.y - 18},
                  {centre.x + 18, centre.y - 18},
                  {centre.x + 18, centre.y + 18},
                  {centre.x - 18, centre.y + 18} };
        c.detections.push_back(d);
    }
    a.v10.status =
        "Cyan: gripper fingers (22.8 x 71 mm). Outer box: spacing needed around each block. Red line: two blocks too close.";
    drawDynamic(c.ui, a);
    drawNavigation(c.ui, a);
    if (!imwrite("chloec10/clearance_preview.png", c.ui)) return 1;
    a.v10.phase = 2;
    a.v10.nextPattern = 3;
    a.v10.status = "Before returning: stack 2/3, expected GREEN. Waiting to see all three colours in order.";
    drawDynamic(c.ui, a);
    drawNavigation(c.ui, a);
    return imwrite("chloec10/timed_cycle_preview.png", c.ui) ? 0 : 1;
}

int detectorCaptureTest(const string& path, bool shared = false) {
    vector<cv::String> files;
    cv::glob(path + "/frame_*_raw.png", files, false);
    if (files.empty()) {
        cerr << "No captured raw frames" << endl;
        return 1;
    }
    calibrationui::ColourModel cfg;
    ifstream settings(path + "/settings.csv");
    string line;
    getline(settings, line);
    while (getline(settings, line)) {
        replace(line.begin(), line.end(), ',', ' ');
        istringstream row(line);
        int k, h, t, sa, v, a;
        if (!(row >> k >> h >> t >> sa >> v >> a) || k < 0 || k > 2) return 1;
        cfg.hue[k] = h;
        cfg.tol[k] = t;
        cfg.sat = sa;
        cfg.val = v;
        cfg.minArea = a;
        int style = 0;
        if (row >> style) cfg.staticStyle = style != 0;
    }
    if (shared) {
        cfg.staticStyle = true;
        cfg.sat = 65;
        cfg.val = 60;
        cfg.minArea = 350;
        cfg.hue = { {0, 60, 110} };
        cfg.tol = { {12, 15, 15} };
    }
    int failures = 0;
    for (auto file : files) {
        auto frame = imread(file);
        auto d = calibrationui::detect(frame, cfg);
        array<int, 3> count{ {0, 0, 0} };
        for (auto p : d) ++count[p.colour];
        if (count != array<int, 3>{{3, 3, 3}}) {
            ++failures;
            cerr << file << ": R" << count[0] << " G" << count[1] << " B" << count[2] << endl;
        }
    }
    cout << "Recorded-camera regression: " << files.size() - failures << "/" << files.size()
        << " frames contain exactly 3R/3G/3B" << endl;
    return failures ? 1 : 0;
}
int selectedCaptureTest(const string& path) {
    calibrationui::ColourModel cfg;
    cfg.staticStyle = true;
    cfg.sat = 65;
    cfg.val = 60;
    cfg.minArea = 350;
    cfg.tol = { {12, 15, 15} };
    auto first = imread(path + "/frame_1_raw.png");
    auto initial = calibrationui::detect(first, cfg);
    if (initial.size() != 9) return 1;
    int failures = 0;
    for (size_t object = 0; object < initial.size(); ++object) {
        calibrationui::App c;
        c.test = true;
        c.useStaticDetector = true;
        c.staticCfg = cfg;
        c.selectionTracked = true;
        c.selectionPosition = initial[object].centre;
        c.selectionColour = initial[object].colour;
        c.selected = int(object);
        int ready = 0, firstReady = 0;
        for (int i = 1; i <= 30; ++i) {
            c.frame = imread(path + "/frame_" + to_string(i) + "_raw.png");
            if (c.frame.empty()) return 1;
            c.update();
            if (c.selectStable) {
                ++ready;
                if (!firstReady) firstReady = i;
            }
        }
        cout << "Selected B" << object + 1 << ": first ready frame " << firstReady << ", ready " << ready << "/30; "
            << c.selectedStability.reason << endl;
        if (!ready) ++failures;
    }
    return failures ? 1 : 0;
}
void drawTestControls(App& a) {
    auto& calibration = *a.calibration;
    drawEnlargedHover(calibration.ui, a);
    button(calibration.ui, Rect(865, 700, 185, 30),
        calibration.enlargedValidation ? "MEASURE AREA: ON" : "MEASURE AREA", calibration.enlargedValidation);
    button(calibration.ui, Rect(1060, 700, 185, 30), "APPLY XY CHECKS",
        calibration.enlargedApproved && calibration.enlargedMatches());
    fill(calibration.ui, Rect(855, 733, 410, 20), Scalar(30, 26, 23));
    label(calibration.ui, "Checks: " + to_string(calibration.enlargedPredicted.size()) + " / 9+ | click here to CLEAR",
        Point(865, 748), .36, GOLD);
    fill(calibration.ui, Rect(0, 752, 1280, 126), Scalar(30, 26, 23));
    label(calibration.ui,
        "TEST trim X " + calibrationui::fmt(a.dynamicTrim.x, 1) + " Y " + calibrationui::fmt(a.dynamicTrim.y, 1) +
        " mm",
        Point(20, 784), .42, GOLD);
    const char* trimNames[] = { "X -0.5", "X +0.5", "Y -0.5", "Y +0.5" };
    for (int i = 0; i < 4; ++i) button(calibration.ui, Rect(440 + i * 82, 775, 75, 28), trimNames[i]);
    button(calibration.ui, Rect(770, 775, 88, 28), "RESET");
    {
        const auto& g = v13Allowance();
        auto g3 = [](double v) { string s = calibrationui::fmt(v, 3); return s.rfind("0.", 0) == 0 ? s.substr(1) : s; };
        label(calibration.ui,
            "AL " + g3(g.yMinus) + " | CL R" + g3(g.yMinusFor(0)) + " G" + g3(g.yMinusFor(1)) + " B" + g3(g.yMinusFor(2)) +
            " | CR R" + g3(g.yPlusFor(0)) + " G" + g3(g.yPlusFor(1)) + " B" + g3(g.yPlusFor(2)),
            Point(20, 804), .38, GOLD);
    }
    const char* allowNames[] = { "AL -", "AL +", "CL -", "CL +", "CR -", "CR +" };
    for (int i = 0; i < 6; ++i) button(calibration.ui, Rect(865 + i * 67, 775, 63, 28), allowNames[i]);
    button(calibration.ui, Rect(230, 845, 220, 30), "1 SEND HOVER",
        a.dynamicStep == 0 && calibration.valid && !calibration.enlargedValidation);
    string pickLabel = a.dynamicStep == 2 ? "2 RESEND SAME PICK" : "2 CONFIRM ALIGNED: PICK";
    bool pickReady = supervisedPickReady(a);
    double remaining = testCommandWait(a);
    if ((a.dynamicStep == 1 || a.dynamicStep == 2) && remaining > 0) {
        pickLabel = "2 WAIT " + to_string(int(ceil(remaining))) + " s";
        pickReady = false;
    }
    button(calibration.ui, Rect(460, 845, 220, 30), pickLabel, pickReady);
    string homeLabel = a.dynamicStep == 3 ? "3 RESEND SAME HOME"
        : a.dynamicStep == 1 ? "3 CANCEL + HOME"
        : "3 REPLACE + HOME";
    if (a.dynamicStep && remaining > 0) homeLabel = "3 WAIT " + to_string(int(ceil(remaining))) + " s";
    button(calibration.ui, Rect(690, 845, 220, 30), homeLabel,
        a.dynamicStep >= 1 && a.dynamicStep <= 3 && remaining <= 0);
    button(calibration.ui, Rect(920, 845, 200, 30), "4 VERIFIED HOME", a.dynamicStep == 3);

    if (calibration.valid && !calibration.detections.empty() && (!calibration.selectionMissing || a.dynamicStep)) {
        auto target = a.dynamicStep
            ? a.dynamicXY
            : v13Target(a,
                calibrationui::project(calibration.mapping.h,
                    calibration.detections[calibration.selected].centre),
                calibration.detections[calibration.selected].colour);
        fill(calibration.ui, Rect(855, 653, 410, 28), Scalar(48, 43, 38));
        label(calibration.ui,
            "COMMAND X " + calibrationui::fmt(target.x, 1) + " Y " + calibrationui::fmt(target.y, 1) + " mm",
            Point(865, 674), .42, GOLD);
    }
    fill(calibration.ui, Rect(0, 810, 1280, 30), Scalar(30, 26, 23));
    label(calibration.ui,
        calibration.entering ? "ACTUAL robot X,Y: " + calibration.input + "   Enter = save / Esc = cancel"
        : (calibration.enlargedValidation || calibration.message.find("DETECTION CAPTURE") == 0) ? calibration.message
        : a.dynamicStatus,
        Point(18, 832), .40, GOLD);
}
int enlargedHoverPreview() {
    App a;
    calibrationui::App c;
    a.calibration = &c;
    a.dynamicTestMode = true;
    a.calibrationMode = true;
    a.enlargedHover = true;
    c.opt.source = "demo";
    for (int i = 0; i < 25; ++i) {
        c.frame = calibrationui::synthetic(c);
        c.update();
    }
    if (!calibrationui::makeMapping(c.grid, c.refs, c.frame.size(), c.mapping)) return 1;
    c.valid = true;
    c.test = true;
    c.enlargedHoverView = true;
    c.frame = calibrationui::synthetic(c);
    c.update();
    c.draw();
    drawTestControls(a);
    drawNavigation(c.ui, a);
    return imwrite("qa_outputs/enlarged_hover_preview.png", c.ui) ? 0 : 1;
}
int uiAuditSelfTest() {
    try {
        App a;
        calibrationui::App c;
        a.calibration = &c;
        c.frame = Mat(480, 640, CV_8UC3, Scalar(120, 120, 120));
        a.lastFrame = c.frame;
        if (!requestAppExit(a)) throw runtime_error("Idle UI exit blocked");
        if (a.finalTopFallback) throw runtime_error("Fallback enabled by default");
        if (!toggleStackVerification(a) || !a.finalTopFallback) throw runtime_error("Fallback selection failed");
        a.v10.phase = 2;
        if (toggleStackVerification(a) || !a.finalTopFallback)
            throw runtime_error("Active cycle changed verification policy");
        a.v10.phase = 0;
        toggleStackVerification(a);
        a.sentCount = 1;
        if (requestAppExit(a)) throw runtime_error("Static exit guard missing");
        a.sentCount = 0;
        for (int stage = 1; stage <= 3; ++stage) {
            a.dynamicStep = stage;
            if (requestAppExit(a)) throw runtime_error("Test exit guard missing");
        }
        a.dynamicStep = 0;
        for (int phase = 1; phase <= 5; ++phase) {
            a.v10.phase = phase;
            if (requestAppExit(a)) throw runtime_error("Dynamic exit guard missing");
        }
        a.v10.phase = 0;
        a.commandLatched = true;
        if (requestAppExit(a)) throw runtime_error("Latched command exit guard missing");
        a.commandLatched = false;

        for (int mode = 1; mode <= 4; ++mode) {
            if (binary8(225 + mode) != string(mode == 1 ? "11100010"
                : mode == 2 ? "11100011"
                : mode == 3 ? "11100100"
                : "11100101"))
                throw runtime_error("Mode byte encoding");
            if (byteDescription(225 + mode).find("MODE ") == string::npos)
                throw runtime_error("Mode byte diagnostic missing");
        }
        auto click = [&](int x, int y) { mouse(EVENT_LBUTTONDOWN, x, y, 0, &a); };
        a.stackVision.configured = true;
        a.stackVision.clearStable = true;
        a.stackVision.emptyStable = true;
        a.stackVision.note = "Stack area empty";
        a.stackVision.lastFrame = chrono::steady_clock::now() - chrono::seconds(2);
        if (stackStartReason(a).find("fresh workspace") == string::npos)
            throw runtime_error("Stale clear/empty check gave misleading reason");
        a.stackVision.lastFrame = chrono::steady_clock::now();
        if (!stackStartReason(a).empty()) throw runtime_error("Fresh clear/empty view blocked Start");
        a.stackVision = StackVision{};
        // Exercise the actual top-level mouse routes, not just their helper functions.
        click(360, 60);
        if (a.pendingControllerMode != 4) throw runtime_error("Calibration mode request");
        if (!a.calibrationMode || a.dynamicTestMode) throw runtime_error("Calibration tab route");
        c.entering = true;
        click(210, 60);
        if (a.dynamicOperate || !c.entering) throw runtime_error("Dynamic bypassed coordinate entry");
        c.entering = false;
        c.diagnosticRemaining = 5;
        click(210, 60);
        if (a.dynamicOperate) throw runtime_error("Dynamic bypassed diagnostic capture");
        c.diagnosticRemaining = 0;
        click(210, 60);
        if (a.pendingControllerMode != 2) throw runtime_error("Dynamic mode request");
        if (!a.dynamicOperate || a.dynamicTestMode || !c.test) throw runtime_error("Dynamic tab route");
        click(1150, 170);
        if (a.selected != 3) throw runtime_error("Dynamic pattern selection");
        click(1100, 495);
        if (!a.v10.allowRepeats) throw runtime_error("Dynamic repeats route");
        click(680, 60);
        if (a.dynamicOperate || a.dynamicTestMode || !a.calibrationMode)
            throw runtime_error("Calibration setup header route");
        click(510, 60);
        if (a.pendingControllerMode != 3) throw runtime_error("Test mode request");
        if (!a.dynamicTestMode) throw runtime_error("Test tab route");
        c.valid = true;
        click(700, 82);
        if (!c.valid) throw runtime_error("Hidden legacy header active");
        click(1000, 785);
        click(225, 785);
        click(30, 855);
        click(1140, 855);
        if (a.dynamicRequest) throw runtime_error("Invisible legacy Test control active");
        click(680, 60);
        if (!a.enlargedHover) throw runtime_error("Enlarged hover toggle");
        click(480, 785);
        if (abs(a.dynamicTrim.x + .5) > 1e-9) throw runtime_error("X trim route");
        click(810, 785);
        if (norm(a.dynamicTrim) > 1e-9) throw runtime_error("Reset trim route");
        a.dynamicRequest = 1;
        click(210, 60);
        if (a.dynamicOperate) throw runtime_error("Pending command bypassed navigation lock");
        a.dynamicRequest = 0;
        a.dynamicStep = 1;
        click(210, 60);
        if (a.dynamicOperate) throw runtime_error("Test session bypassed navigation lock");
        a.dynamicStep = 0;
        click(210, 60);
        a.v10.phase = 3;
        int running = a.selected;
        click(880, 170);
        if (a.selected != running || a.v10.nextPattern != 1)
            throw runtime_error("Active pattern changed instead of queue");
        click(30, 60);
        if (!a.dynamicOperate) throw runtime_error("Active cycle allowed Static navigation");
        click(880, 645);
        if (a.v10.phase != 5) throw runtime_error("Pause PC did not enter recovery");
        // Simulate restart into Static with an unresolved journal.
        a.dynamicOperate = false;
        a.calibrationMode = false;
        a.dynamicTestMode = false;
        a.pendingControllerMode = 0;
        int recoveryTx = a.lastSent;
        int recoveryToken = a.v10.token;
        click(210, 60);
        if (!a.dynamicOperate || !a.calibrationMode || a.dynamicTestMode || a.v10.phase != 5 ||
            a.pendingControllerMode || a.lastSent != recoveryTx || a.v10.token != recoveryToken)
            throw runtime_error("Restart recovery page inaccessible or requested a robot command");
        click(30, 60);
        if (!a.dynamicOperate || a.v10.phase != 5) throw runtime_error("Recovery allowed Static mode change");
        a.v10.phase = 0;
        a.v10.nextPattern = 0;
        click(30, 60);
        if (a.calibrationMode || a.dynamicOperate) throw runtime_error("Static tab route");
        calibrationui::App fitted;
        fitted.opt.source = "demo";
        for (int n = 0; n < 25; ++n) {
            fitted.frame = calibrationui::synthetic(fitted);
            fitted.update();
        }
        if (!calibrationui::makeMapping(fitted.grid, fitted.refs, fitted.frame.size(), fitted.mapping))
            throw runtime_error("Persistence fixture fit");
        fitted.savePersistentCalibration();
        calibrationui::App reload;
        if (!reload.loadPersistentCalibration()) throw runtime_error("Saved fit reload");
        reload.invalidate("Camera moved");
        calibrationui::App invalidated;
        if (invalidated.loadPersistentCalibration()) throw runtime_error("Invalidated fit restored after restart");
        fitted.savePersistentCalibration();
        if (!invalidated.loadPersistentCalibration()) throw runtime_error("Refit did not restore persistence");
        {
            calibrationui::ColourModel model;
            calibrationui::DetectionFilter filter;
            Mat seed(180, 360, CV_8UC3, Scalar(0, 0, 150)), dim = seed.clone();
            for (int k = 0; k < 3; ++k) {
                rectangle(seed, Rect(25 + k * 110, 60, 40, 40),
                    Scalar(k == 0 ? 0
                        : k == 1 ? 60
                        : 110,
                        200, 160),
                    FILLED);
                rectangle(dim, Rect(25 + k * 110, 60, 40, 40), Scalar(k == 0 ? 0 : k == 1 ? 60 : 110, 45, 30), FILLED);
            }
            cvtColor(seed, seed, COLOR_HSV2BGR);
            cvtColor(dim, dim, COLOR_HSV2BGR);
            auto raw = calibrationui::detect(seed, model);
            if (raw.size() != 3) throw runtime_error("Reacquisition fixture seed");
            filter.update(raw);
            auto global = calibrationui::detect(dim, model);
            if (!global.empty()) throw runtime_error("Dim fixture should need local reacquisition");
            auto rescued = filter.reacquire(dim, model, global);
            if (rescued.size() != 3) throw runtime_error("Current-frame dim reacquisition failed");
            Mat empty(seed.size(), CV_8UC3, Scalar(150, 150, 150));
            if (!filter.reacquire(empty, model, {}).empty())
                throw runtime_error("Reacquisition invented a missing block");
            Mat shifted;
            warpAffine(dim, shifted, Mat(Matx23d(1, 0, 20, 0, 1, 0)), dim.size(), INTER_NEAREST, BORDER_CONSTANT,
                Scalar(150, 150, 150));
            if (!filter.reacquire(shifted, model, {}).empty()) throw runtime_error("Reacquisition concealed movement");
        }
        {
            StackVision monitor;
            monitor.reference = Mat(300, 420, CV_8UC3, Scalar(130, 130, 130));
            monitor.workspace = Rect(5, 5, 410, 290);
            monitor.stack = Rect(290, 20, 110, 110);
            monitor.configured = true;
            monitor.faceArea = 900;
            auto t = chrono::steady_clock::now();
            Mat exposure(300, 420, CV_8UC3, Scalar(148, 148, 148));
            for (int n = 0; n < 8; ++n) monitor.observe(exposure, {}, t + chrono::milliseconds(n * 120));
            if (!monitor.startReady(t + chrono::milliseconds(840)))
                throw runtime_error("Uniform exposure change blocked clear workspace");
            Mat blocked = exposure.clone();
            rectangle(blocked, Rect(80, 140, 60, 60), Scalar(60, 60, 60), FILLED);
            monitor.observe(blocked, {}, t + chrono::milliseconds(960));
            if (monitor.clearStable) throw runtime_error("Localized neutral obstruction accepted");
            rectangle(blocked, Rect(80, 140, 60, 60), Scalar(0, 0, 230), FILLED);
            monitor.observe(blocked, {}, t + chrono::milliseconds(1080));
            if (monitor.clearStable || monitor.clearReason.find("Red/orange") == string::npos)
                throw runtime_error("Arm-colour obstruction accepted or unexplained");
            monitor.savedReference = true;
            monitor.captureMapping = calibrationui::H(.5, 0, 0, 0, .5, 0, 0, 0, 1);
            monitor.faceArea = 400;
            calibrationui::Detection d;
            d.colour = 1;
            d.area = 74 * 74;
            d.box = Rect(70, 140, 74, 74);
            d.quad = { {70, 140}, {144, 140}, {144, 214}, {70, 214} };
            d.centre = { 107, 177 };
            Mat pickup = monitor.reference.clone();
            rectangle(pickup, d.box, Scalar(0, 180, 0), FILLED);
            for (int n = 0; n < 8; ++n) monitor.observe(pickup, { d }, t + chrono::milliseconds(1200 + n * 120));
            if (!monitor.clearStable) throw runtime_error("Perspective-correct pickup masking failed");
        }
        cout
            << "PASS workspace monitor: exposure shift, localized/arm obstruction rejection and mapped block-size masking"
            << endl;
        cout << "PASS current-image local reacquisition: dim RGB, removed block and moved block rejection" << endl;
        cout
            << "PASS UI mouse routes: four tabs, entry/diagnostic/pending/session locks, patterns/repeats, hidden targets, hover/trim, queued pattern and pause recovery"
            << endl;
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
int savedReferenceSelfTest() {
    try {
        App a;
        a.referenceDynamic = true;
        calibrationui::App c;
        a.calibration = &c;
        c.valid = true;
        c.mapping.h = calibrationui::H(0, 1, 200, 1, 0, -300, 0, 0, 1);
        a.lastFrame = Mat(480, 640, CV_8UC3, Scalar(120, 120, 120));
        c.frame = a.lastFrame;
        c.mapping.resolution = a.lastFrame.size();
        teachStackView(a);
        if (!a.stackVision.configured || a.v10.emptyBackground.empty())
            throw runtime_error("Capture failed: " + a.message);
        App restored;
        restored.referenceDynamic = true;
        restored.calibration = &c;
        loadStackView(restored);
        if (!restored.stackVision.configured || restored.v10.emptyBackground.empty() || !restored.v10.workspaceChecked)
            throw runtime_error("Restart did not restore reference");
        v10SelectReturnArea(restored, false);
        if (restored.v10.emptyBackground.empty() || !restored.v10.workspaceChecked)
            throw runtime_error("Original area lost full-area reference");
        v10SelectReturnArea(restored, true);
        if (restored.v10.emptyBackground.empty()) throw runtime_error("Enlarged area lost reference");
        selectVisionProfile(restored, false);
        if (restored.stackVision.configured) throw runtime_error("Dynamic reference leaked into Static");
        restored.lastFrame = Mat(480, 640, CV_8UC3, Scalar(135, 135, 135));
        teachStackView(restored);
        if (!restored.stackVision.configured) throw runtime_error("Static profile capture failed");
        selectVisionProfile(restored, true);
        if (!restored.stackVision.configured || restored.stackVision.reference.at<Vec3b>(0, 0)[0] != 120 ||
            restored.v10.emptyBackground.at<Vec3b>(0, 0)[0] != 120)
            throw runtime_error("Static capture overwrote Dynamic reference");
        selectVisionProfile(restored, false);
        if (restored.stackVision.reference.at<Vec3b>(0, 0)[0] != 135)
            throw runtime_error("Static reference did not survive mode round trip");
        selectVisionProfile(restored, true);
        c.mapping.h(0, 2) += 1;
        updateStackVision(restored, a.lastFrame);
        if (restored.stackVision.configured || !restored.v10.emptyBackground.empty())
            throw runtime_error("Mapping change retained reference");
        App mismatch;
        mismatch.referenceDynamic = true;
        mismatch.calibration = &c;
        loadStackView(mismatch);
        if (mismatch.stackVision.configured) throw runtime_error("Changed mapping loaded old reference");
        c.mapping.h(0, 2) -= 1;
        loadStackView(mismatch);
        updateStackVision(mismatch, Mat(240, 320, CV_8UC3, Scalar(120, 120, 120)));
        if (mismatch.stackVision.configured || !mismatch.v10.emptyBackground.empty())
            throw runtime_error("Resolution change retained reference");
        cout << "Saved clear reference: capture, restart, area reuse, mapping and resolution invalidation PASS" << endl;
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
int fitReviewPreview() {
    App a;
    calibrationui::App c;
    a.calibration = &c;
    a.calibrationMode = true;
    c.opt.source = "demo";
    for (int i = 0; i < 25; ++i) {
        c.frame = calibrationui::synthetic(c);
        c.update();
    }
    c.labelsOK = c.refsOK = c.planeOK = true;
    c.refs[1].x += 10;
    c.calibration();
    if (!c.fitReview || c.lastFitMax <= c.opt.fitLimit || c.valid) return 1;
    for (int i = 0; i < 9; ++i)
        if (abs(norm(c.fitDelta[i]) - c.fitResidual[i]) > 1e-9) return 1;
    c.draw();
    drawNavigation(c.ui, a);
    return imwrite("fit_review_preview.png", c.ui) ? 0 : 1;
}
int responsiveVisionSelfTest() {
    try {
        auto t = chrono::steady_clock::now();
        ReturnEvidence e;
        for (int f = 0; f < 3; ++f) e.update(true, t + chrono::milliseconds(f * 40));
        if (e.ready(t + chrono::milliseconds(80))) throw runtime_error("Single/three sightings confirmed a return");
        e.update(true, t + chrono::milliseconds(120));
        if (!e.ready(t + chrono::milliseconds(120))) throw runtime_error("Four return sightings did not confirm");
        if (e.ready(t + chrono::seconds(2))) throw runtime_error("Stale returns accepted");
        e.update(true, t + chrono::seconds(2));
        if (e.ready(t + chrono::seconds(2))) throw runtime_error("One image revived expired returns");
        cout << "PASS responsive returns: four sightings, expiry and fresh reacquisition\n";
        App a;
        calibrationui::App c;
        a.calibration = &c;
        a.dynamicOperate = true;
        c.valid = true;
        c.mapping.h = calibrationui::H(1, 0, 0, 0, 1, 0, 0, 0, 1);
        c.mapping.resolution = { 640, 480 };
        c.frame = Mat(480, 640, CV_8UC3, Scalar(130, 130, 130));
        c.staticCfg = stackColourModel(a);
        c.useStaticDetector = true;
        c.test = true;
        a.v10.drop = { {{100, 100}, {250, 100}, {400, 100}} };
        a.v10.colours = { {0, 1, 2} };
        for (int i = 0; i < 3; ++i)
            rectangle(c.frame, Rect(int(a.v10.drop[i].x) - 18, 82, 37, 37),
                i == 0 ? Scalar(0, 0, 180)
                : i == 1 ? Scalar(0, 180, 0)
                : Scalar(180, 0, 0),
                FILLED);
        // No tracker/global detections: a stationary return must reacquire from current pixels.
        for (int i = 0; i < 3; ++i)
            if (!v10ReturnMatch(a, i).seen) throw runtime_error("Target-local RGB reacquisition failed");
        rectangle(c.frame, Rect(380, 80, 42, 42), Scalar(130, 130, 130), FILLED);
        if (v10ReturnMatch(a, 2).seen) throw runtime_error("Missing return invented from old evidence");
        rectangle(c.frame, Rect(382, 82, 37, 37), Scalar(0, 0, 180), FILLED);
        if (v10ReturnMatch(a, 2).seen) throw runtime_error("Wrong colour confirmed as blue");
        c.frame.setTo(Scalar(130, 130, 130));
        rectangle(c.frame, Rect(402, 82, 37, 37), Scalar(180, 0, 0), FILLED);
        if (v10ReturnMatch(a, 2).seen) throw runtime_error("20mm displaced return accepted");
        c.frame.setTo(Scalar(130, 130, 130));
        rectangle(c.frame, Rect(394, 82, 37, 37), Scalar(180, 0, 0), FILLED);
        if (!v10ReturnMatch(a, 2).seen) throw runtime_error("12mm return settling offset not reacquired");
        c.frame.setTo(Scalar(130, 130, 130));
        for (int i = 0; i < 2; ++i) {
            calibrationui::Detection d;
            d.colour = 2;
            d.centre = { 399.0 + i * 3, 100 };
            c.detections.push_back(d);
        }
        if (v10ReturnMatch(a, 2).seen) throw runtime_error("Ambiguous same-colour returns accepted");
        cout
            << "PASS responsive returns: real RGB ROI reacquisition, 12mm offset, missing/wrong/20mm/ambiguous rejection\n";
        a.selected = 1;
        array<Point2d, 3> points{ {{100, 100}, {250, 100}, {400, 100}} };
        for (int f = 0; f < 5; ++f) v10UpdatePickupSamples(a.v10, true, points, t + chrono::milliseconds(f * 40));
        if (!v10PickupSamplesReady(a.v10, points, t + chrono::milliseconds(160)))
            throw runtime_error("Five-frame pickup readiness failed");
        array<Point2d, 3> median;
        if (!v10MedianPickup(a.v10, median) || median != points)
            throw runtime_error("Start median still required20 samples or changed targets");
        auto original = points;
        auto old = calibrationui::PATTERNS[0];
        for (int pattern : {2, 3, 1}) {
            v10SelectPattern(a, pattern);
            array<Point2d, 3> reordered;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (calibrationui::PATTERNS[pattern - 1][2 - i] == old[2 - j]) reordered[i] = original[j];
            if (!v10PickupSamplesReady(a.v10, reordered, t + chrono::milliseconds(160)) || a.v10.stableFrames != 5)
                throw runtime_error("Pattern switch lost or misassociated pickup readings");
        }
        v10SetRepeatMode(a, false);
        if (a.v10.stableFrames != 5) throw runtime_error("Repeat toggle discarded matching readings");
        auto shifted = points;
        shifted[0].x += 20;
        if (v10PickupSamplesReady(a.v10, shifted, t + chrono::milliseconds(160)))
            throw runtime_error("Changed source reused old readings");
        if (v10PickupSamplesReady(a.v10, points, t + chrono::seconds(3))) throw runtime_error("Stale pickup accepted");
        cout << "PASS responsive pickup: five frames, pattern reorder, repeat retention, moved/stale rejection\n";
        V10State slow;
        array<Point2d, 3> slowPoints{ {{350, -120}, {430, -40}, {500, -120}} };
        for (int f = 0; f < 12; ++f) {
            auto when = t + chrono::milliseconds(f * 600);
            auto jitter = slowPoints;
            for (auto& point : jitter) point.x += (f % 2 ? .5 : -.5);
            v10UpdatePickupSamples(slow, true, jitter, when);
            if (f >= 4 && (!v10PickupSamplesReady(slow, jitter, when) || slow.stableFrames < 5))
                throw runtime_error("Continuous slow-camera evidence fell from5 to4 or lost Ready");
        }
        V10State fast;
        for (int f = 0; f < 80; ++f) {
            auto when = t + chrono::milliseconds(f * 10);
            v10UpdatePickupSamples(fast, true, slowPoints, when);
            if (f >= 16 && !v10PickupSamplesReady(fast, slowPoints, when))
                throw runtime_error("Fast camera displayed5/5 but could never become Ready");
        }
        auto slowNow = t + chrono::milliseconds(6600);
        auto retained = slow.stableFrames;
        v10UpdatePickupSamples(slow, false, {}, slowNow + chrono::milliseconds(300));
        if (slow.stableFrames != retained) throw runtime_error("Brief dropout restarted stable readings");
        v10UpdatePickupSamples(slow, true, slowPoints, slowNow + chrono::milliseconds(600));
        if (!v10PickupSamplesReady(slow, slowPoints, slowNow + chrono::milliseconds(600)))
            throw runtime_error("Matching reacquisition recounted readings");
        auto changed = slowPoints;
        changed[0].x += 20;
        if (v10PickupSamplesReady(slow, changed, slowNow + chrono::milliseconds(600)))
            throw runtime_error("Stable latch allowed moved pickup");
        for (int f = 0; f < 3; ++f)
            v10UpdatePickupSamples(slow, true, changed, slowNow + chrono::milliseconds(700 + f * 100));
        if (slow.stableFrames != 1) throw runtime_error("Confirmed movement failed to reset stable readings");
        v10UpdatePickupSamples(slow, false, {}, slowNow + chrono::seconds(3));
        if (slow.stableFrames) throw runtime_error("Long loss retained pickup latch");
        App request;
        calibrationui::App requestCamera;
        request.calibration = &requestCamera;
        request.dynamicOperate = true;
        requestCamera.valid = true;
        requestCamera.opt.source = "camera";
        requestCamera.frame = Mat(480, 640, CV_8UC3, Scalar(130, 130, 130));
        int previousTx = request.lastSent;
        v10RequestStart(request);
        if (!request.v10.startRequested || request.v10.phase || request.lastSent != previousTx)
            throw runtime_error("Start click sent before fresh-frame validation");
        v10Tick(request);
        if (!request.v10.startRequested || request.lastSent != previousTx ||
            request.v10.status.find("START WAIT:") != 0)
            throw runtime_error("Blocked Start silently disappeared or transmitted");
        request.v10.startDeadline = chrono::steady_clock::now() - chrono::seconds(1);
        v10Tick(request);
        if (request.v10.startRequested || request.lastSent != previousTx ||
            request.v10.status.find("START NOT SENT:") != 0)
            throw runtime_error("Expired request remained armed or lost its reason");
        v10RequestStart(request);
        v10SetRepeatMode(request, false);
        if (request.v10.startRequested) throw runtime_error("Preference change retained pending Start");
        v10RequestStart(request);
        requestCamera.freeze = true;
        v10Tick(request);
        if (request.v10.startRequested || request.lastSent != previousTx)
            throw runtime_error("Frozen camera retained Start or sent motion");
        App ready;
        calibrationui::App readyCamera;
        ready.calibration = &readyCamera;
        ready.dynamicOperate = true;
        ready.calibrationMode = true;
        ready.serialConnected = true;
        // No port is opened. The transport must fail only AFTER the real Start validation path.
        readyCamera.valid = true;
        readyCamera.opt.source = "camera";
        readyCamera.mapping.h = calibrationui::H(1, 0, 200, 0, 1, -300, 0, 0, 1);
        readyCamera.frame = Mat(480, 640, CV_8UC3, Scalar(130, 130, 130));
        readyCamera.mapping.resolution = readyCamera.frame.size();
        ready.v10.workspaceChecked = true;
        ready.v10.emptyBackground = readyCamera.frame.clone();
        ready.v10.backgroundMapping = readyCamera.mapping.h;
        ready.v10.backgroundRange = v10ActiveBounds(ready);
        for (int i = 0; i < 3; ++i) {
            calibrationui::Detection d;
            d.colour = i;
            d.centre = calibrationui::project(readyCamera.mapping.h.inv(), slowPoints[i]);
            for (auto off : vector<Point>{ {-17, -17}, {17, -17}, {17, 17}, {-17, 17} })
                d.quad.push_back(Point(cvRound(d.centre.x), cvRound(d.centre.y)) + off);
            d.box = boundingRect(d.quad);
            d.area = 35 * 35;
            readyCamera.detections.push_back(d);
        }
        array<Point2d, 3> readyPlan;
        array<int, 3> readyColours;
        if (!v10PickPlan(ready, readyPlan, readyColours)) throw runtime_error("Real Start fixture pickup plan failed");
        auto readyTime = chrono::steady_clock::now();
        for (int f = 0; f < 5; ++f)
            v10UpdatePickupSamples(ready.v10, true, readyPlan, readyTime - chrono::milliseconds((4 - f) * 40));
        ready.stackVision.configured = true;
        ready.stackVision.clearStable = true;
        ready.stackVision.emptyStable = true;
        ready.stackVision.lastFrame = readyTime;
        auto readiness = v10ReadyReason(ready);
        if (!readiness.empty()) throw runtime_error("Five-sample real Start fixture blocked: " + readiness);
        int readyTx = ready.lastSent;
        v10RequestStart(ready);
        v10Tick(ready);
        if (ready.v10.startRequested || ready.v10.phase != 5 || ready.v10.status.find("TX failed") != 0 ||
            ready.lastSent != readyTx)
            throw runtime_error("Queued Ready click failed real median/validation dispatch path: " + ready.v10.status);
        cout
            << "PASS stable Start: slow/100fps cameras, jitter/dropouts, real movement/loss, fresh-frame click, blocking/expiry/cancellation and actual five-sample Start dispatch without a serial port\n";
        StackVision v;
        v.configured = true;
        v.workspace = { 10, 10, 600, 400 };
        v.stack = { 430, 70, 150, 150 };
        v.faceArea = 37 * 37;
        v.reference = Mat(480, 640, CV_8UC3, Scalar(130, 130, 130));
        v.begin({ {2, 0, 1} });
        Mat placed = v.reference.clone();
        rectangle(placed, { 470, 110, 37, 37 }, Scalar(180, 0, 0), FILLED);
        calibrationui::ColourModel model;
        model.staticStyle = true;
        model.minArea = 350;
        auto objects = calibrationui::detect(placed, model);
        v.observe(placed, objects, t);
        if (v.progress) throw runtime_error("One frame committed observed layer");
        v.observe(placed, objects, t + chrono::milliseconds(60));
        if (v.progress != 1 || v.observedLayers[0] != 2)
            throw runtime_error("Two fresh 60ms observations did not update layer");
        cout << "PASS observed layer: two real-image observations in60ms; no one-frame commitment\n";
        auto start = chrono::steady_clock::now();
        for (int f = 0; f < 40; ++f) {
            auto objects = calibrationui::detect(placed, model);
            v.observe(placed, objects, chrono::steady_clock::now());
        }
        cout << "BENCHMARK 640x480 detector+workspace: "
            << chrono::duration<double, milli>(chrono::steady_clock::now() - start).count() / 40
            << " ms/frame (synthetic scene, not live camera FPS)\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
int v11RecoverySelfTest() {
    try {
        calibrationui::ColourModel cfg;
        cfg.staticStyle = false;
        cfg.minArea = 80;
        auto h = calibrationui::H::eye();
        for (bool brightTop : {true, false}) {
            Mat scene(180, 220, CV_8UC3, Scalar(130, 130, 130));
            rectangle(scene, { 60, 60, 35, 35 }, Scalar(brightTop ? 180 : 80, 0, 0), FILLED);
            rectangle(scene, { 95, 60, 10, 35 }, Scalar(brightTop ? 80 : 180, 0, 0), FILLED);
            auto detected = calibrationui::detect(scene, cfg);
            if (detected.size() != 1) throw runtime_error("Cuboid fixture did not produce one merged silhouette");
            auto top = v11TopFace(scene, cfg, h, detected.front());
            if (!top.pickFaceVerified || norm(top.centre - Point2d(77, 77)) > 1.5)
                throw runtime_error("Top/side silhouette still biased the pickup centre");
        }
        Mat ambiguous(180, 220, CV_8UC3, Scalar(130, 130, 130));
        rectangle(ambiguous, { 60, 60, 35, 35 }, Scalar(180, 0, 0), FILLED);
        rectangle(ambiguous, { 95, 60, 35, 35 }, Scalar(80, 0, 0), FILLED);
        calibrationui::Detection merged;
        merged.colour = 2;
        merged.quad = { {60, 60}, {129, 60}, {129, 94}, {60, 94} };
        merged.box = boundingRect(merged.quad);
        merged.centre = calibrationui::crossCentre(merged.quad);
        merged.area = 69 * 34;
        if (v11TopFace(ambiguous, cfg, h, merged).pickFaceVerified)
            throw runtime_error("Ambiguous two-square candidate was guessed");
        Mat plain(180, 220, CV_8UC3, Scalar(130, 130, 130));
        rectangle(plain, { 60, 60, 45, 35 }, Scalar(180, 0, 0), FILLED);
        auto silhouettes = calibrationui::detect(plain, cfg);
        if (silhouettes.empty() || v11TopFace(plain, cfg, h, silhouettes.front()).pickFaceVerified)
            throw runtime_error("Oversize single face accepted without geometry proof");
        cout
            << "PASS V11 top face: bright/dark top with visible side corrected from current pixels; ambiguous and oversize silhouettes blocked\n";
        App a;
        calibrationui::App c;
        a.calibration = &c;
        c.valid = true;
        c.mapping.h = h;
        c.frame = Mat(180, 220, CV_8UC3, Scalar(130, 130, 130));
        a.v10.emptyBackground = c.frame.clone();
        a.v10.pick[0] = { 80, 80 };
        if (!v11PickupEmptyNow(a, 0)) throw runtime_error("Actual empty pickup patch rejected");
        rectangle(c.frame, { 65, 65, 30, 30 }, Scalar(20, 20, 20), FILLED);
        if (v11PickupEmptyNow(a, 0)) throw runtime_error("Occluded/missing detection was mistaken for removal");
        c.frame = a.v10.emptyBackground.clone();
        calibrationui::Detection d;
        d.centre = { 80, 80 };
        c.detections = { d };
        if (v11PickupEmptyNow(a, 0)) throw runtime_error("Visible source counted as removed");
        auto fields = v11ControlFields(17, 4, 3);
        if (fields != vector<uint8_t>({ 17, 4, 3, 25 })) throw runtime_error("Checkpoint checksum/token encoding");
        bool rejected = false;
        try {
            v11ControlFields(0, 4, 3);
        }
        catch (...) {
            rejected = true;
        }
        if (!rejected) throw runtime_error("Invalid checkpoint token accepted");
        calibrationui::Detection face;
        face.quad = { {60, 60}, {94, 60}, {94, 94}, {60, 94} };
        face.pickFaceVerified = true;
        if (!v11GripperOpeningFits(face, h, { 77, 77 }) || v11GripperOpeningFits(face, h, { 77, 83 }))
            throw runtime_error("Measured finger opening failed centered/offset edge check");
        cout
            << "PASS V11 removal evidence: real empty patch required; arm/visible source blocked; token/checksum controls; physical opening margin\n";
        a.dynamicOperate = true;
        a.v10.token = 17;
        a.v10.phase = 1;
        int prior = a.lastSent;
        a.busy = true;
        v11OperatorRequest(a, 2);
        if (a.v10.phase != 5 || a.v10.operatorRequest != 2 || a.lastSent != prior)
            throw runtime_error("HOME click during TX sent/interleaved bytes instead of queuing");
        a.busy = false;
        v11OperatorRequest(a, 3);
        if (!a.v10.resetArmed || a.v10.resetAwait)
            throw runtime_error("Reset bypassed physical recovery acknowledgement");
        v11OperatorRequest(a, 3);
        if (!a.v10.resetAwait || a.v10.phase != 5)
            throw runtime_error("Reset cleared PC before controller confirmation");
        v11OperatorRequest(a, 3);
        if (a.v10.phase != 5 || !a.v10.resetAwait)
            throw runtime_error("PC reset before controller reset packet was sent");
        a.v10.resetPacketSent = true;
        v11OperatorRequest(a, 3);
        if (a.v10.phase || a.v10.resetAwait || a.lastSent != prior)
            throw runtime_error("Acknowledged PC reset sent motion or retained stale cycle");
        cout
            << "PASS V11 operator controls: busy-time request queued, no blind motion, staged recovery/reset acknowledgement\n";
        calibrationui::App faces;
        faces.opt.source = "demo";
        faces.frame = Mat(300, 300, CV_8UC3, Scalar(130, 130, 130));
        faces.refs.clear();
        faces.cfg.staticStyle = false;
        faces.cfg.val = 35;
        faces.cfg.sat = 55;
        faces.cfg.minArea = 80;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                int x = 40 + col * 90, y = 40 + row * 90 - col * 3, k = (row * 3 + col) % 3;
                Scalar bright = k == 0 ? Scalar(0, 0, 200) : k == 1 ? Scalar(0, 200, 0) : Scalar(200, 0, 0);
                Scalar dark = k == 0 ? Scalar(0, 0, 80) : k == 1 ? Scalar(0, 80, 0) : Scalar(80, 0, 0);
                rectangle(faces.frame, { x, y, 35, 35 }, bright, FILLED);
                rectangle(faces.frame, { x + 35, y, 10, 35 }, dark, FILLED);
                faces.refs.push_back({ double(x + 17), double(y + 17) });
            }
        for (int frame = 0; frame < 25; ++frame) faces.update();
        calibrationui::Mapping fitted;
        if (!calibrationui::makeMapping(faces.grid, faces.refs, faces.frame.size(), fitted))
            throw runtime_error("Side-visible nine-point calibration failed");
        for (size_t i = 0; i < faces.grid.size(); ++i)
            if (!faces.grid[i].pickFaceVerified || norm(faces.grid[i].centre - faces.refs[i]) > 1.5)
                throw runtime_error("Calibration still used silhouette centre instead of pickup top centre");
        faces.mapping = fitted;
        faces.valid = true;
        faces.v11TopFaceCalibration = false;
        faces.savePersistentCalibration();
        calibrationui::App legacy;
        if (!legacy.loadPersistentCalibration() || legacy.v11TopFaceCalibration)
            throw runtime_error("Legacy calibration incorrectly marked as top-face calibrated");
        App gate;
        gate.calibration = &legacy;
        if (v10ReadyReason(gate).find("Top-face estimator changed") == string::npos)
            throw runtime_error("Legacy centre calibration allowed automatic pickup");
        faces.v11TopFaceCalibration = true;
        faces.savePersistentCalibration();
        calibrationui::App current;
        if (!current.loadPersistentCalibration() || !current.v11TopFaceCalibration)
            throw runtime_error("New top-face calibration marker lost");
        cout
            << "PASS V11 calibration: visible sides excluded consistently at all nine reference points; legacy estimator blocked; new estimator persisted\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
int lightingReferenceSelfTest() {
    try {
        App a;
        calibrationui::App c;
        a.calibration = &c;
        a.referenceDynamic = true;
        c.valid = true;
        c.opt.source = "camera";
        c.mapping.h = calibrationui::H(0, 1, 200, 1, 0, -300, 0, 0, 1);
        c.mapping.resolution = { 640, 480 };
        auto mapping = c.mapping.h;
        int tx = a.lastSent;
        auto capture = [&](int profile, int brightness) {
            c.frame = Mat(480, 640, CV_8UC3, Scalar(brightness, brightness, brightness));
            a.lastFrame = c.frame.clone();
            beginLightingCapture(a, profile);
            if (!a.lightCaptureActive) throw runtime_error("Capture did not start: " + a.message);
            auto t = a.lightCaptureSince;
            for (int f = 0; f < 12; ++f) tickLightingCapture(a, t + chrono::milliseconds(f * 50));
            if (a.lightCaptureActive || !a.stackVision.configured ||
                abs(mean(a.stackVision.reference)[0] - brightness) > .1)
                throw runtime_error("Stable capture failed: " + a.message);
            if (a.lastSent != tx || norm(Mat(mapping) - Mat(c.mapping.h)) > 1e-8)
                throw runtime_error("Lighting capture changed XY or sent a robot command");
            };
        capture(0, 160);
        if (!selectLightingProfile(a, 1) || a.stackVision.configured || !a.v10.emptyBackground.empty())
            throw runtime_error("Missing OFF profile silently reused ON reference");
        capture(1, 80);
        selectLightingProfile(a, 0);
        if (!a.stackVision.configured || mean(a.stackVision.reference)[0] != 160 ||
            mean(a.v10.emptyBackground)[0] != 160)
            throw runtime_error("OFF capture overwrote ON reference");
        // The screenshot's brightness mismatch must not be hidden by widening clearance thresholds.
        auto time = chrono::steady_clock::now();
        c.frame = Mat(480, 640, CV_8UC3, Scalar(80, 80, 80));
        for (int f = 0; f < 16; ++f) a.stackVision.observe(c.frame, {}, time + chrono::milliseconds(f * 50));
        if (a.stackVision.clearStable) throw runtime_error("Wrong lighting reference accepted as clear");
        selectLightingProfile(a, 1);
        time = chrono::steady_clock::now();
        for (int f = 0; f < 16; ++f) a.stackVision.observe(c.frame, {}, time + chrono::milliseconds(f * 50));
        if (!a.stackVision.clearStable || !a.stackVision.emptyStable)
            throw runtime_error("Correct OFF reference did not clear stable empty view");
        a.v10.phase = 1;
        if (selectLightingProfile(a, 0) || a.lightProfile != 1)
            throw runtime_error("Lighting profile changed during active cycle");
        beginLightingCapture(a, 0);
        if (a.lightCaptureActive) throw runtime_error("Capture started during robot cycle");
        a.v10.phase = 0;
        // Parked objects outside the actual workspace must not prevent capture.
        c.frame = Mat(480, 640, CV_8UC3, Scalar(80, 80, 80));
        rectangle(c.frame, { 580, 400, 35, 35 }, Scalar(0, 0, 200), FILLED);
        a.lastFrame = c.frame.clone();
        beginLightingCapture(a, 1);
        tickLightingCapture(a);
        if (!a.lightCaptureActive || a.lightCaptureCount != 1)
            throw runtime_error("Outside-workspace object unnecessarily blocked lighting capture");
        cancelLightingCapture(a, "test outside cancel");
        c.frame = Mat(480, 640, CV_8UC3, Scalar(80, 80, 80));
        // Empty-reference capture may not absorb a visible cube.
        rectangle(c.frame, { 120, 120, 45, 45 }, Scalar(0, 0, 200), FILLED);
        a.lastFrame = c.frame.clone();
        beginLightingCapture(a, 1);
        tickLightingCapture(a);
        if (a.lightCaptureActive || mean(imread(visionImage(a)))[0] != 80)
            throw runtime_error("Occupied capture replaced saved empty reference");
        c.frame = Mat(480, 640, CV_8UC3, Scalar(80, 80, 80));
        a.lastFrame = c.frame.clone();
        beginLightingCapture(a, 1);
        tickLightingCapture(a);
        rectangle(c.frame, { 100, 100, 120, 120 }, Scalar(35, 35, 35), FILLED);
        tickLightingCapture(a);
        if (!a.lightCaptureActive || a.lightCaptureCount != 0)
            throw runtime_error("Moving/settling image was averaged into background");
        cancelLightingCapture(a, "test cancel");
        if (mean(imread(visionImage(a)))[0] != 80) throw runtime_error("Cancelled capture damaged prior reference");
        selectVisionProfile(a, false);
        capture(0, 200);
        capture(1, 100);
        selectVisionProfile(a, true);
        if (mean(a.stackVision.reference)[0] != 80) throw runtime_error("Static OFF leaked into Dynamic OFF");
        selectLightingProfile(a, 0);
        if (mean(a.stackVision.reference)[0] != 160) throw runtime_error("Static ON leaked into Dynamic ON");
        selectLightingProfile(a, 1);
        App restart;
        restart.referenceDynamic = true;
        restart.calibration = &c;
        loadLightingPreference(restart);
        loadStackView(restart);
        if (restart.lightProfile != 1 || !restart.stackVision.configured ||
            mean(restart.stackVision.reference)[0] != 80)
            throw runtime_error("Lighting selection/reference did not survive restart");
        c.mapping.h(0, 2) += 1;
        App moved;
        moved.referenceDynamic = true;
        moved.lightProfile = 1;
        moved.calibration = &c;
        loadStackView(moved);
        if (moved.stackVision.configured) throw runtime_error("Changed mapping reused old lighting reference");
        c.mapping.h = mapping;
        // Exercise actual mouse routes for capture setup and the selected light condition.
        App ui;
        ui.calibration = &c;
        ui.calibrationMode = true;
        ui.referenceDynamic = true;
        ui.lastFrame = c.frame.clone();
        mouse(EVENT_LBUTTONDOWN, 1100, 60, 0, &ui);
        if (!ui.lightingSetupOpen) throw runtime_error("Lighting setup header route failed");
        mouse(EVENT_LBUTTONDOWN, 1100, 225, 0, &ui);
        if (ui.lightProfile != 1) throw runtime_error("Lighting OFF selector route failed");
        mouse(EVENT_LBUTTONDOWN, 1000, 565, 0, &ui);
        if (ui.lightingSetupOpen) throw runtime_error("Lighting setup close route failed");
        c.frame = Mat(480, 640, CV_8UC3, Scalar(80, 80, 80));
        a.lastFrame = c.frame.clone();
        beginLightingCapture(a, 1);
        auto noisyTime = a.lightCaptureSince;
        RNG noiseGenerator(20261021);
        for (int f = 0; f < 12; ++f) {
            Mat noisy(c.frame.size(), CV_32FC3);
            noiseGenerator.fill(noisy, RNG::NORMAL, Scalar::all(80), Scalar::all(8));
            noisy.convertTo(c.frame, CV_8UC3);
            a.lastFrame = c.frame.clone();
            tickLightingCapture(a, noisyTime + chrono::milliseconds(f * 50));
        }
        if (a.lightCaptureActive || !a.stackVision.configured || abs(mean(a.stackVision.reference)[0] - 80) > 1)
            throw runtime_error("Low-light sensor noise prevented stable capture");
        a.lightingSetupOpen = false;
        selectLightingProfile(a, 0);
        auto autoTime = chrono::steady_clock::now() + chrono::seconds(3);
        Mat off(480, 640, CV_8UC3, Scalar(80, 80, 80));
        for (int i = 0; i < 8; ++i) tickAutomaticLighting(a, off, autoTime + chrono::milliseconds(i * 150));
        if (a.lightProfile != 1 || !a.autoLightReady) throw runtime_error("Automatic OFF matching failed");
        // V13: LIGHT OFF camera auto-exposure / white-balance drift (16 levels) must still match OFF.
        Mat drifted(480, 640, CV_8UC3, Scalar(96, 94, 98));
        for (int i = 0; i < 8; ++i) tickAutomaticLighting(a, drifted, autoTime + chrono::milliseconds(1200 + i * 100));
        if (a.lightProfile != 1 || !a.autoLightReady || a.autoLightComp[1] > lightMatchLimit() || a.autoLightRaw[1] <= lightMatchLimit())
            throw runtime_error("LIGHT OFF camera drift blocked automatic matching");
        cout << "PASS V13 lighting: 16-level LIGHT OFF camera drift matches OFF (raw " << a.autoLightRaw[1]
            << ", compensated " << a.autoLightComp[1] << "); a 55-level change stays unknown\n";
        Mat unknown(480, 640, CV_8UC3, Scalar(25, 25, 25));
        tickAutomaticLighting(a, unknown, autoTime + chrono::seconds(2));
        if (a.autoLightReady || stackStartReason(a).find("Lighting:") != 0)
            throw runtime_error("Unknown lighting allowed Start");
        Mat on(480, 640, CV_8UC3, Scalar(160, 160, 160));
        a.v10.phase = 1;
        for (int i = 0; i < 8; ++i)
            tickAutomaticLighting(a, on, autoTime + chrono::seconds(3) + chrono::milliseconds(i * 150));
        if (a.lightProfile != 1) throw runtime_error("Automatic lighting switched during cycle");
        a.v10.phase = 0;
        rectangle(on, { 130, 130, 30, 30 }, Scalar(0, 0, 220), FILLED);
        for (int i = 0; i < 8; ++i)
            tickAutomaticLighting(a, on, autoTime + chrono::seconds(5) + chrono::milliseconds(i * 150));
        if (a.lightProfile != 0 || !a.autoLightReady)
            throw runtime_error("Automatic ON matching with small block failed");
        tickAutomaticLighting(a, off, autoTime + chrono::seconds(7));
        tickAutomaticLighting(a, on, autoTime + chrono::seconds(7) + chrono::milliseconds(150));
        if (a.lightProfile != 0 || a.autoLightReady) throw runtime_error("Transient lighting bypassed settling");
        cout
            << "PASS automatic lighting: stable ON/OFF, small block tolerance, unknown Start gate, active-cycle lock and transient rejection\n";
        cout
            << "PASS lighting: independent ON/OFF and Static/Dynamic references;12-image capture; missing-profile/occupancy/motion/session guards; matching brightness clearance; restart/mapping protection; UI routes; no serial or XY changes\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
int lightingSetupPreview() {
    App a;
    calibrationui::App c;
    a.calibration = &c;
    a.calibrationMode = true;
    a.referenceDynamic = true;
    a.lightingSetupOpen = true;
    c.frame = Mat(480, 640, CV_8UC3, Scalar(130, 130, 130));
    c.draw();
    drawNavigation(c.ui, a);
    drawLightingSetup(c.ui, a);
    return imwrite("lighting_setup_preview.png", c.ui) ? 0 : 1;
}
int scenePairSelfTest(const string& onPath, const string& offPath) {
    try {
        vector<Mat> images;
        for (auto path : vector<string>{ onPath, offPath }) {
            Mat im = imread(path);
            if (im.empty()) throw runtime_error("Cannot decode scene image: " + path);
            // The supplied regressions are camera-window screenshots with a30px title bar.
            // Raw camera images need no crop; this adjustment is test-only.
            if (im.cols == 639 && im.rows >= 507 && im.rows <= 508) im = im(Rect(0, 30, im.cols, im.rows - 30)).clone();
            images.push_back(im);
        }
        calibrationui::App c;
        c.opt.source = "image";
        c.frame = images[0];
        for (int i = 0; i < 25; ++i) c.update();
        if (c.grid.size() != 9 || !c.gridQualityReady ||
            !calibrationui::makeMapping(c.grid, c.refs, c.frame.size(), c.mapping) || c.mapping.max > 2)
            throw runtime_error("ON scene did not pass the normal nine-point calibration gates");
        for (auto d : c.grid)
            if (!d.pickFaceVerified) throw runtime_error("ON calibration face rejected");
        cout << "PASS supplied ON calibration: nine verified faces; max " << c.mapping.max
            << " mm (unchanged2mm gate)\n";
        c.valid = true;
        c.v11TopFaceCalibration = true;
        c.test = true;
        c.useStaticDetector = true;
        c.staticCfg = c.cfg;
        vector<Point2d> onCentres;
        for (int image = 0; image < 2; ++image) {
            c.frame = images[image];
            // Cropped screenshots differ by one bottom row; preserve H exactly.
            c.mapping.resolution = c.frame.size();
            c.update();
            auto faces = c.detections;
            if (!calibrationui::gridOrder(faces, false, false))
                throw runtime_error("Runtime scene did not detect nine separate blocks");
            array<int, 3> counts{ {0, 0, 0} };
            double drift = 0;
            for (size_t i = 0; i < faces.size(); ++i) {
                auto face = v11TopFace(c.frame, c.activeColourModel(), c.mapping.h, faces[i]);
                auto xy = calibrationui::project(c.mapping.h, face.centre);
                if (!face.pickFaceVerified || !v11GripperOpeningFits(face, c.mapping.h, xy))
                    throw runtime_error("Runtime face or physical35mm opening check failed at P" + to_string(i + 1));
                ++counts[face.colour];
                if (!image)
                    onCentres.push_back(xy);
                else
                    drift = max(drift, norm(xy - onCentres[i]));
            }
            double centreBudget = GRIPPER_INNER_MM * .5 - CLEARANCE_MARGIN_MM - 17.5;
            cout << "Cross-light centre difference " << drift << " mm; physical35mm cube centre allowance "
                << centreBudget << " mm\n";
            if (counts != array<int, 3>{{3, 3, 3}} || drift > centreBudget)
                throw runtime_error(
                    "Lighting changed colour identity or exceeded the physical gripper centre allowance");
            cout << "PASS supplied " << (image ? "OFF" : "ON")
                << " runtime:9/9 verified using SAME ON mapping; max cross-light centre difference " << drift
                << " mm\n";
        }
        Mat absent = images[1].clone();
        rectangle(absent, { 185, 70, 70, 70 }, Scalar(130, 130, 130), FILLED);
        auto missing = calibrationui::detect(absent, c.staticCfg);
        if (missing.size() >= 9) throw runtime_error("Removed block was filled in by detector");
        cout << "PASS real-image removed-block rejection: no inferred or retained block\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
int smoothCycleSelfTest() {
    try {
        auto t = chrono::steady_clock::now();
        ReturnEvidence slow;
        for (int i = 0; i < 6; ++i) slow.update(true, t + chrono::seconds(i));
        if (!slow.ready(t + chrono::seconds(5))) throw runtime_error("1fps evidence never became ready");
        if (slow.ready(t + chrono::seconds(9))) throw runtime_error("Stalled camera evidence remained ready");
        slow.update(true, t + chrono::seconds(10));
        if (slow.ready(t + chrono::seconds(10))) throw runtime_error("One image revived evidence after pause");
        for (int i = 11; i < 16; ++i) slow.update(true, t + chrono::seconds(i));
        if (!slow.ready(t + chrono::seconds(15))) throw runtime_error("Fresh slow-camera reacquisition failed");
        cout
            << "PASS slow-camera evidence:1fps confirmation, stale expiry, no one-frame revival, fresh reacquisition\n";
        App a;
        calibrationui::App c;
        a.calibration = &c;
        c.valid = true;
        c.opt.source = "camera";
        c.mapping.h = calibrationui::H::eye();
        c.frame = Mat(300, 300, CV_8UC3, Scalar(130, 130, 130));
        a.v10.pick = { {{80, 80}, {160, 80}, {80, 160}} };
        a.v10.emptyBackground = c.frame.clone();
        rectangle(c.frame, { 65, 65, 30, 30 }, Scalar(110, 110, 110), FILLED);
        // A moderate neutral background change must not invalidate a clearly observed placement.
        rectangle(c.frame, { 65, 65, 30, 30 }, Scalar(80, 80, 80), FILLED);
        a.stackVision.configured = true;
        a.stackVision.progress = 2;
        a.stackVision.expected = { {2, 1, 0} };
        a.stackVision.top = 1;
        a.stackVision.topClean = a.stackVision.clearStable = true;
        a.stackVision.lastFrame = t;
        if (v11PickupEmptyNow(a, 0) || !v12PickupVerified(a, 0, t))
            throw runtime_error("Verified layer blocked by a neutral source-background change");
        calibrationui::Detection source;
        source.centre = { 80, 80 };
        c.detections = { source };
        if (v12PickupVerified(a, 0, t)) throw runtime_error("Visible source bypassed placement verification");
        c.detections.clear();
        a.stackVision.clearStable = false;
        if (v12PickupVerified(a, 0, t)) throw runtime_error("Occluded workspace bypassed source verification");
        a.stackVision.clearStable = true;
        a.stackVision.top = 0;
        if (v12PickupVerified(a, 0, t)) throw runtime_error("Wrong current stack top bypassed verification");
        a.stackVision.top = 1;
        if (v12PickupVerified(a, 0, t + chrono::seconds(2)))
            throw runtime_error("Stale image bypassed placement verification");
        a.v10.phase = 1;
        a.v10.handshakeStep = 1;
        a.v10.handshakeSince = t - chrono::seconds(120);
        a.stackVision.progress = 0;
        v11StackHandshake(a, t);
        if (a.v10.phase != 1 || a.v10.retryControl || a.v10.operatorRequest)
            throw runtime_error("Vision wait automatically latched HOLD");
        a.dynamicOperate = true;
        a.v10.phase = 3;
        a.v10.handshakeStep = 4;
        a.v10.returnObserved = true;
        a.v10.lastReturnTx = t;
        a.v10.handshakeSince = t - chrono::seconds(120);
        a.v10.drop = { {{80, 80}, {160, 80}, {80, 160}} };
        v10Tick(a);
        if (a.v10.phase != 3 || a.v10.retryControl || a.v10.operatorRequest)
            throw runtime_error("Return wait automatically latched HOLD");
        cout
            << "PASS checkpoint decisions: positive placement survives background change; visible source/occlusion/wrong top/stale image blocked;120s waits send no HOLD\n";
        array<Point2d, 3> xy{ {{400, -40}, {450, 40}, {500, -40}} };
        auto stack = serialFrameSequence(220, v10Fields(17, 2, xy, false), 221);
        auto ret = serialFrameSequence(222, v10Fields(17, 2, xy, true), 223);
        auto permit = serialFrameSequence(234, v11ControlFields(17, 4, 2), 235);
        if ((stack.size() - 1) * SERIAL_SYMBOL_MS + 2 * SERIAL_SYMBOL_MS != 9000 ||
            (ret.size() - 1) * SERIAL_SYMBOL_MS != 8000 || (permit.size() - 1) * SERIAL_SYMBOL_MS != 3000)
            throw runtime_error("Packet timing budget changed");
        for (auto sequence : vector<vector<uint8_t>>{ stack, ret, permit })
            for (int phase = 0; phase < 20; ++phase) {
                int time = phase;
                vector<uint8_t> decoded;
                auto valueAt = [&](int when) {
                    return sequence[min(sequence.size() - 1, size_t(when / SERIAL_SYMBOL_MS))];
                    };
                for (size_t symbol = 1; symbol < sequence.size(); symbol += 2) {
                    while (valueAt(time) != 0) time += 20;
                    time += 40;
                    if (valueAt(time) != 0) throw runtime_error("Zero separator too short for RAPID stable check");
                    while (valueAt(time) == 0) time += 20;
                    auto value = valueAt(time);
                    time += 40;
                    if (valueAt(time) != value) throw runtime_error("Byte too short for RAPID stable check");
                    decoded.push_back(value);
                }
                vector<uint8_t> expected;
                for (size_t i = 1; i < sequence.size(); i += 2) expected.push_back(sequence[i]);
                if (decoded != expected) throw runtime_error("RAPID polling model lost or duplicated a byte");
            }
        vector<uint8_t> written;
        atomic<size_t> sent{ 0 };
        atomic<int> error{ -1 };
        atomic<bool> done{ false };
        auto began = chrono::steady_clock::now();
        thread worker([&]() {
            writeTimedSymbols(
                permit, 10,
                [&](uint8_t value) {
                    written.push_back(value);
                    return true;
                },
                sent, error, done);
            });
        // A slow UI/camera iteration must not be the symbol clock.
        this_thread::sleep_for(chrono::milliseconds(250));
        bool finishedIndependently = done.load();
        worker.join();
        if (!finishedIndependently || written != permit || error.load() != -1 || sent.load() != permit.size())
            throw runtime_error("Camera stall stretched or corrupted background transmission");
        atomic<size_t> partial{ 0 };
        atomic<int> failed{ -1 };
        atomic<bool> stopped{ false };
        int calls = 0;
        writeTimedSymbols(permit, 0, [&](uint8_t) { return ++calls < 4; }, partial, failed, stopped);
        if (failed.load() != 3 || partial.load() != 3 || calls != 4 || !stopped.load())
            throw runtime_error("Failed write continued with a partial packet");
        (void)began;
        cout
            << "PASS serial worker: packet order retained during a slow camera iteration; partial write stops immediately\n";
        cout
            << "PASS wire protocol: unchanged tokens/coordinates/checksum;250ms symbols decoded at all20ms poll phases; nominal stack9s/return8s/permit3s\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
}
// ---- V13 scenario self-test
// Synthetic scenes (world = pixel + (300,-200) mm). Times are simulated; frames are captured, not sent.
int v13SelfTest(const string& onPath, const string& offPath) {
    using clock = chrono::steady_clock;
    auto fail = [](const string& why) { throw runtime_error(why); };
    // The scenario tests use the shared centre-line gain; per-colour defaults are tested in section 12.
    v13Allowance().colourYMinus = { {NAN, NAN, NAN} };
    v13Allowance().colourYPlus = { {NAN, NAN, NAN} };
    try {
        // 1. Frame encoding matches the MOD decoder exactly.
        {
            auto f = v13TargetFields(17, 1, 2, 5, { 412.5, -96 });
            if (f.size() != 9) fail("V13 field count");
            for (auto b : f)
                if (!b) fail("V13 zero field byte");
            int sum = 0;
            for (int i = 0; i < 8; ++i) sum += f[i];
            if (f[8] != sum % 251 + 1) fail("V13 checksum");
            double x = 100 + ((f[4] - 1) * 128 + f[5] - 1) / 2.0, y = -400 + ((f[6] - 1) * 128 + f[7] - 1) / 2.0;
            if (abs(x - 412.5) > 1e-9 || abs(y + 96) > 1e-9 || f[4] > 16 || f[5] > 128 || f[6] > 16 || f[7] > 128)
                fail("V13 XY does not decode with the MOD formula");
            for (auto bad : vector<array<int, 4>>{ {{17, 3, 2, 5}},
                                                  {{17, 1, 7, 5}},
                                                  {{17, 1, 2, 0}},
                                                  {{17, 2, 3, 5}},
                                                  {{17, 2, 6, 5}},
                                                  {{0, 1, 2, 5}} }) {
                bool rejected = false;
                try {
                    v13TargetFields(bad[0], bad[1], bad[2], bad[3], { 412.5, -96 });
                }
                catch (...) {
                    rejected = true;
                }
                if (!rejected) fail("Invalid V13 fields accepted");
            }
            auto seq = serialFrameSequence(236, f, 237);
            if (seq.size() != 23 || (seq.size() - 1) * SERIAL_SYMBOL_MS != 5500) fail("V13 timing budget");
            cout
                << "PASS V13 frame: 9 non-zero fields, MOD XY decode, checksum, range/mode/step rejection, 5.5 s on the wire\n";
        }
        {
            V10State v;
            auto t = clock::now();
            v.phase = 3;
            v.handshakeStep = 4;
            v.lastReturnTx = t - chrono::seconds(31);
            v.v13.hint = "confirming what the camera sees";
            v.v13.pendingAttempt = 7;
            if (!v13ReturnHandoffDue(v, t)) fail("Camera hint/pending re-pick starved return handoff");
            v.lastReturnTx = t;
            if (v13ReturnHandoffDue(v, t)) fail("Return handoff resent too soon");
            v.lastReturnTx = t - chrono::seconds(31);
            v.returnConfirmed[2] = true;
            if (v13ReturnHandoffDue(v, t)) fail("Handoff replay after observed return");
            v.returnConfirmed[2] = false;
            v.handshakeStep = 5;
            if (v13ReturnHandoffDue(v, t)) fail("Handoff replay during later return");
            cout << "PASS return handoff: hints cannot starve retry; active/later returns excluded\n";
        }
        vector<vector<uint8_t>> sent;
        serialTestCapture = &sent;
        // Logic tests use raw camera positions: offsets off here, restored on exit.
        struct Restore {
            V13Allowance keep = v13Allowance();
            ~Restore() {
                serialTestCapture = nullptr;
                v13Allowance() = keep;
            }
        } restore;
        v13Allowance().blueY = 0;
        v13Allowance().yMinus = 0;
        auto block = [](int colour, Point2d world) {
            calibrationui::Detection d;
            d.colour = colour;
            Point2d px = world - Point2d(300, -200);
            d.centre = px;
            d.quad = { Point(cvRound(px.x - 17), cvRound(px.y - 17)), Point(cvRound(px.x + 17), cvRound(px.y - 17)),
                      Point(cvRound(px.x + 17), cvRound(px.y + 17)), Point(cvRound(px.x - 17), cvRound(px.y + 17)) };
            d.box = boundingRect(d.quad);
            d.area = 34 * 34;
            d.pickFaceVerified = true;
            return d;
            };
        auto setup = [&](App& a, calibrationui::App& c) {
            a.calibration = &c;
            a.dynamicOperate = true;
            c.valid = true;
            c.opt.source = "camera";
            c.mapping.h = calibrationui::H(1, 0, 300, 0, 1, -200, 0, 0, 1);
            c.mapping.robot = calibrationui::nominalRobot();
            // re-pick logic tests: a calibration box covering the whole synthetic area (allowance 0)
            for (auto& q : c.mapping.robot)
                q = Point2d(q.x < 400 ? 300 : q.x > 480 ? 620 : 460, q.y < -50 ? -300 : q.y > 30 ? 120 : -90);
            c.frame = Mat(480, 640, CV_8UC3, Scalar(130, 130, 130));
            a.v10.emptyBackground = c.frame.clone();
            a.v10.backgroundMapping = c.mapping.h;
            a.v10.enlargedReturns = true;
            a.v10.token = 9;
            a.v10.colours = { {2, 0, 1} };
            a.v10.pick = { {{420, -100}, {480, 40}, {420, 40}} };
            auto& sv = a.stackVision;
            sv.configured = true;
            sv.stack = Rect(18, 331, 110, 110);
            sv.workspace = Rect(0, 0, 640, 480);
            sv.expected = a.v10.colours;
            sv.armed = true;
            };
        auto frames236 = [&]() {
            vector<vector<uint8_t>> out;
            for (auto& q : sent)
                if (q.size() > 2 && q[1] == 236) {
                    vector<uint8_t> fields;
                    for (size_t i = 3; i + 3 < q.size(); i += 2) fields.push_back(q[i]);
                    out.push_back(fields);
                }
            return out;
            };
        auto decode = [](const vector<uint8_t>& f) {
            return Point2d(100 + ((f[4] - 1) * 128 + f[5] - 1) / 2.0, -400 + ((f[6] - 1) * 128 + f[7] - 1) / 2.0);
            };
        // Drive the stack supervisor at 10 Hz; the arm obstructs the view between armFrom and armTo seconds.
        auto runStack = [&](App& a, clock::time_point t0, double from, double to, double armFrom, double armTo) {
            a.v10.v13.basePending.fill(false); // flat synthetic frames: face-evidence tests only
            for (double s = from; s <= to + 1e-9; s += .1) {
                auto t = t0 + chrono::milliseconds(int(s * 1000));
                a.stackVision.lastFrame = t;
                a.stackVision.clearStable = !(s >= armFrom && s < armTo);
                a.stackVision.clearFrames = a.stackVision.clearStable ? 5 : 0;
                v13Observe(a, t);
                v13StackSupervise(a, t);
                if (a.v10.phase == 5) return;
            }
            };
        // 2. Missed pick, block nudged 8.9 mm: camera finds it again and sends ONE re-pick at the new position.
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 1;
            a.v10.handshakeStep = 1;
            c.detections = { block(2, {412, -96}), block(0, {480, 40}), block(1, {420, 40}) };
            auto& sv = a.stackVision;
            sv.progress = 0;
            sv.emptyStable = true;
            sv.topCount = 0;
            sv.currentFace = -1;
            auto t0 = clock::now() + chrono::hours(1);
            v13Release(a, 1, t0);
            runStack(a, t0, 0, 9.5, 2, 8);
            if (!frames236().empty()) fail("Re-pick sent before the transfer could have finished");
            runStack(a, t0, 9.6, 14, -1, -1);
            auto f = frames236();
            if (f.size() != 1) fail("Expected exactly one re-pick, got " + to_string(f.size()));
            if (f[0][1] != 1 || f[0][2] != 1 || f[0][3] != 1 || norm(decode(f[0]) - Point2d(412, -96)) > .01 ||
                norm(a.v10.pick[0] - Point2d(412, -96)) > .01 || a.v10.v13.tries[1] != 1)
                fail("Re-pick fields/target wrong");
            if (a.v10.phase != 1 || a.v10.retryControl) fail("Missed pick escalated to HOLD");
            // Robot busy: no arm seen for 7 s -> identical resend (same attempt id, so the MOD cannot move twice).
            runStack(a, t0, 14.1, 22, -1, -1);
            f = frames236();
            if (f.size() < 2 || f[1] != f[0] || a.v10.v13.tries[1] != 1) fail("Pending re-pick not resent identically");
            size_t before = f.size();
            // Robot executes: arm visible, then the layer appears on the stack. No further re-pick.
            runStack(a, t0, 22.1, 26, 22.1, 26);
            sv.topCount = 1;
            sv.currentFace = 2;
            sv.emptyStable = false;
            runStack(a, t0, 26.1, 45, -1, -1);
            if (frames236().size() != before) fail("Re-pick repeated after the layer was placed");
            cout
                << "PASS V13 missed pick: no decision during the transfer, one re-pick at the camera position (8.9 mm away), identical resend while unexecuted, silent once placed, no HOLD\n";
        }
        // 3. Missed block vanished (may be in the gripper): HOLD, and never pick a different pre-existing block.
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 1;
            a.v10.handshakeStep = 1;
            a.v10.allowRepeats = true;
            c.detections = { block(2, {420, -100}), block(2, {360, -200}), block(0, {480, 40}), block(1, {420, 40}) };
            auto& sv = a.stackVision;
            sv.emptyStable = true;
            auto t0 = clock::now() + chrono::hours(1);
            v13Release(a, 1, t0);
            c.detections.erase(c.detections.begin()); // the gripped block is gone from view
            runStack(a, t0, 0, 15, 2, 8);
            if (!frames236().empty() || a.v10.phase == 5) fail("Vanished block: acted before the search window ended");
            runStack(a, t0, 15.1, 25, -1, -1);
            if (!frames236().empty())
                fail("Re-picked a different pre-existing block while the missed one may be in the gripper");
            if (a.v10.phase != 5 || a.v10.retryControl != 1) fail("Vanished block did not request HOLD");
            cout
                << "PASS V13 vanished block: searches 8 s, then HOLD (possibly in gripper); never substitutes another block\n";
        }
        // 3b. Missed block knocked far (new position) is still accounted for and re-picked.
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 1;
            a.v10.handshakeStep = 1;
            c.detections = { block(2, {420, -100}), block(0, {480, 40}), block(1, {420, 40}) };
            a.stackVision.emptyStable = true;
            auto t0 = clock::now() + chrono::hours(1);
            v13Release(a, 1, t0);
            c.detections[0] = block(2, { 360, -200 });
            runStack(a, t0, 0, 15, 2, 8);
            auto f = frames236();
            if (f.size() != 1 || norm(decode(f[0]) - Point2d(360, -200)) > .01)
                fail("Knocked block at a new position not re-picked");
            cout
                << "PASS V13 knocked block: found 116 mm away at a position that was empty at release, re-picked there\n";
        }
        // 3c. Missed block landed on top of another block: HOLD (height unknown, MOD picks at table height).
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 1;
            a.v10.handshakeStep = 1;
            c.detections = { block(2, {420, -100}), block(0, {360, -200}), block(0, {480, 40}), block(1, {420, 40}) };
            a.stackVision.emptyStable = true;
            auto t0 = clock::now() + chrono::hours(1);
            v13Release(a, 1, t0);
            c.detections.erase(c.detections.begin(), c.detections.begin() + 2);
            c.detections.push_back(block(2, { 362, -196 }));
            runStack(a, t0, 0, 16, 2, 8);
            if (!frames236().empty() || a.v10.phase != 5) fail("Block resting on another block was not held");
            cout << "PASS V13 block on another block: HOLD, no re-pick at table height\n";
        }
        // 3d. Frozen camera feed is detected; live sensor noise is not.
        {
            V13State r;
            Mat still(480, 640, CV_8UC3, Scalar(120, 120, 120));
            auto t0 = clock::now();
            for (int i = 0; i <= 20; ++i) v13UpdateFrozen(r, still, t0 + chrono::milliseconds(i * 100));
            if (!r.frozen) fail("Frozen feed not detected");
            V13State live;
            RNG rng(7);
            for (int i = 0; i <= 30; ++i) {
                Mat noisy = still.clone();
                Mat n(noisy.size(), CV_8UC3);
                rng.fill(n, RNG::UNIFORM, 0, 3);
                noisy += n;
                v13UpdateFrozen(live, noisy, t0 + chrono::milliseconds(i * 100));
            }
            if (live.frozen) fail("Live noisy feed flagged as frozen");
            cout
                << "PASS V13 frozen camera: identical frames for 1.5 s block decisions; normal sensor noise does not\n";
        }
        // 4. Serious: stack knocked over after layer 1, and repeated misses.
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 1;
            a.v10.handshakeStep = 1;
            c.detections = { block(0, {480, 40}), block(1, {420, 40}) };
            auto& sv = a.stackVision;
            sv.progress = 1;
            sv.emptyStable = true;
            auto t0 = clock::now() + chrono::hours(1);
            v13Release(a, 2, t0);
            runStack(a, t0, 0, 16, 2, 8);
            if (a.v10.phase == 5) fail("HOLD before the disturbance persisted");
            runStack(a, t0, 16.1, 20, -1, -1);
            if (a.v10.phase != 5 || a.v10.retryControl != 1 || a.v10.v13.attempt)
                fail("Knocked stack did not request HOLD");
            App b;
            calibrationui::App d;
            setup(b, d);
            b.v10.phase = 1;
            b.v10.handshakeStep = 1;
            d.detections = { block(2, {412, -96}), block(0, {480, 40}), block(1, {420, 40}) };
            b.stackVision.emptyStable = true;
            b.v10.v13.tries[1] = V13_MAX_TRIES;
            v13Release(b, 1, t0);
            runStack(b, t0, 0, 14, 2, 8);
            if (b.v10.phase != 5 || b.v10.retryControl != 1) fail("Third failed re-pick did not request HOLD");
            cout
                << "PASS V13 serious states: knocked stack (persisting 6 s) and 3 failed re-picks request HOLD; nothing else does\n";
        }
        // 5. Next block bumped before its turn: the permit carries the corrected pickup (no blind miss).
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 1;
            a.v10.handshakeStep = 1;
            a.v10.pick[0] = { 412, -96 };
            c.detections = { block(0, {492, 40}), block(1, {420, 40}) };
            auto& sv = a.stackVision;
            sv.progress = 1;
            sv.topClean = true;
            sv.top = 2;
            sv.clearStable = true;
            sv.topCount = 1;
            sv.currentFace = 2;
            auto t0 = clock::now() + chrono::hours(1);
            for (double s = 0; s <= 3; s += .1) {
                auto t = t0 + chrono::milliseconds(int(s * 1000));
                sv.lastFrame = t;
                v11StackHandshake(a, t);
            }
            auto f = frames236();
            bool permit = false;
            for (auto& q : sent)
                if (q.size() > 2 && q[1] == 234) permit = true;
            if (f.empty() || permit || f[0][1] != 2 || f[0][2] != 1 || norm(decode(f[0]) - Point2d(492, 40)) > .01 ||
                norm(a.v10.pick[1] - Point2d(492, 40)) > .01)
                fail("Bumped next block not retargeted");
            for (auto& q : f)
                if (q != f[0]) fail("Retarget resend changed attempt or XY");
            if (a.v10.v13.releasedStep != 2) fail("Retarget did not release towards checkpoint 2");
            App b;
            calibrationui::App d;
            setup(b, d);
            b.v10.phase = 1;
            b.v10.handshakeStep = 1;
            b.v10.pick[0] = { 412, -96 };
            d.detections = { block(0, {480, 40}), block(1, {420, 40}) };
            b.stackVision = sv;
            sent.clear();
            for (double s = 0; s <= 1.5; s += .1) {
                auto t = t0 + chrono::milliseconds(int(s * 1000));
                b.stackVision.lastFrame = t;
                v11StackHandshake(b, t);
            }
            if (!frames236().empty() || sent.empty() || sent[0][1] != 234)
                fail("Unmoved next block did not get a normal permit");
            cout
                << "PASS V13 retarget: a 12 mm bump of the next block is sent with the permit (identical resends); unmoved blocks get the normal V12 permit\n";
        }
        // 6. Return: block landed elsewhere / lost / still on the stack.
        {
            auto returnSetup = [&](App& a, calibrationui::App& c) {
                setup(a, c);
                a.v10.phase = 3;
                a.v10.handshakeStep = 4;
                a.v10.drop = { {{350, -220}, {520, -40}, {470, -150}} };
                a.stackVision.progress = 3;
                };
            auto runReturn = [&](App& a, clock::time_point t0, double from, double to) {
                a.v10.v13.basePending.fill(false);
                for (double s = from; s <= to + 1e-9; s += .1) {
                    auto t = t0 + chrono::milliseconds(int(s * 1000));
                    a.stackVision.lastFrame = t;
                    a.stackVision.clearStable = !(s >= 2 && s < 8);
                    a.stackVision.clearFrames = a.stackVision.clearStable ? 5 : 0;
                    v13Observe(a, t);
                    v13ReturnSupervise(a, t);
                    if (a.v10.phase == 5) return;
                }
                };
            auto t0 = clock::now() + chrono::hours(1);
            sent.clear();
            App a;
            calibrationui::App c;
            returnSetup(a, c);
            c.detections = { block(0, {350, -220}) };
            v13Release(a, 4, t0);
            a.stackVision.topCount = 1;
            a.stackVision.currentFace = 0;
            c.detections.push_back(block(1, { 430, -60 }));
            runReturn(a, t0, 0, 14);
            if (norm(a.v10.drop[2] - Point2d(430, -60)) > .01 || !v10ReturnMatch(a, 2).seen || !frames236().empty())
                fail("Dropped return block not accepted where it landed");
            App b;
            calibrationui::App d;
            returnSetup(b, d);
            d.detections = { block(0, {350, -220}) };
            v13Release(b, 4, t0);
            b.stackVision.topCount = 1;
            b.stackVision.currentFace = 0;
            runReturn(b, t0, 0, 24);
            if (b.v10.v13.lost[2] || b.v10.phase != 5 || b.v10.retryControl != 1)
                fail("Unseen return block did not request HOLD");
            sent.clear();
            App e;
            calibrationui::App g;
            returnSetup(e, g);
            g.detections = { block(0, {350, -220}) };
            v13Release(e, 4, t0);
            e.stackVision.topCount = 1;
            e.stackVision.currentFace = 1;
            runReturn(e, t0, 0, 14);
            auto f = frames236();
            if (f.size() != 1 || f[0][1] != 1 || f[0][2] != 4 || norm(decode(f[0]) - Point2d(470, -150)) > .01)
                fail("Return pick miss not re-picked from the stack");
            cout
                << "PASS V13 return: landed-elsewhere block accepted at its camera position, unseen block -> HOLD (may be in gripper), missed stack pick re-picked to a free position\n";
        }
        // 6b. V13.5 lost frame: the return plan never reached the controller (robot stays at READY, arm never
        //     seen). The PC resends the identical 222 frame instead of a re-pick the robot would ignore.
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 3;
            a.v10.handshakeStep = 4;
            a.selected = 2;
            a.v10.drop = { {{350, -220}, {520, -40}, {470, -150}} };
            a.stackVision.progress = 3;
            c.detections = { block(0, {350, -220}) };
            a.v10.v13.releasedStep = 0;
            v10Send(a, true);
            auto plan = sent.back();
            if (a.v10.v13.releasedStep != 4 || a.v10.v13.releaseStart != 222)
                fail("Return plan did not record its release frame");
            a.stackVision.topCount = 1;
            a.stackVision.currentFace = 1;
            a.v10.v13.basePending.fill(false);
            sent.clear();
            auto t0 = clock::now() + chrono::hours(1);
            a.v10.v13.releasedAt = t0;
            int resent = 0;
            double firstAt = -1;
            for (double s = 0; s <= 90; s += .1) {
                auto t = t0 + chrono::milliseconds(int(s * 1000));
                a.stackVision.lastFrame = t;
                a.stackVision.clearStable = true;
                a.stackVision.clearFrames = 5;
                size_t before = sent.size();
                v13Observe(a, t);
                v13ReturnSupervise(a, t);
                if (a.v10.phase == 5) break;
                for (size_t i = before; i < sent.size(); ++i)
                    if (sent[i] == plan) {
                        ++resent;
                        if (firstAt < 0) firstAt = s;
                    }
                if (!frames236().empty()) break;
            }
            if (resent != V13_MAX_RELEASE_RESENDS || firstAt < V13_ARMLESS_S || firstAt > V13_ARMLESS_S + 3)
                fail("Lost return plan was not resent identically (" + to_string(resent) + " at " + to_string(firstAt) +
                    " s)");
            auto f = frames236();
            if (f.size() != 1 || f[0][1] != 1 || f[0][2] != 4)
                fail("After the resends a still-unchanged stack did not fall back to a re-pick");
            // Arm seen = the robot did move: a missed grip goes straight to the re-pick, no plan resend.
            sent.clear();
            App b;
            calibrationui::App d;
            setup(b, d);
            b.v10.phase = 3;
            b.v10.handshakeStep = 4;
            b.selected = 2;
            b.v10.drop = a.v10.drop;
            b.stackVision.progress = 3;
            d.detections = { block(0, {350, -220}) };
            b.v10.v13.releasedStep = 0;
            v10Send(b, true);
            plan = sent.back();
            sent.clear();
            b.stackVision.topCount = 1;
            b.stackVision.currentFace = 1;
            b.v10.v13.basePending.fill(false);
            b.v10.v13.releasedAt = t0;
            for (double s = 0; s <= 20; s += .1) {
                auto t = t0 + chrono::milliseconds(int(s * 1000));
                b.stackVision.lastFrame = t;
                b.stackVision.clearStable = !(s >= 2 && s < 8);
                b.stackVision.clearFrames = b.stackVision.clearStable ? 5 : 0;
                v13Observe(b, t);
                v13ReturnSupervise(b, t);
                if (!frames236().empty()) break;
            }
            for (auto& q : sent)
                if (q == plan) fail("Plan resent although the arm was seen");
            if (frames236().size() != 1) fail("Missed return grip with arm seen was not re-picked");
            // Lost permit at the stack: the identical permit for the previous step is resent.
            sent.clear();
            App e;
            calibrationui::App g;
            setup(e, g);
            e.v10.phase = 1;
            e.v10.handshakeStep = 2;
            g.detections = { block(0, {480, 40}), block(1, {420, 40}) };
            e.stackVision.progress = 1;
            e.stackVision.emptyStable = false;
            e.stackVision.topCount = 1;
            e.stackVision.currentFace = 2;
            v11SendControl(e, 4, 1);
            auto permit = sent.back();
            sent.clear();
            e.v10.v13.basePending.fill(false);
            e.v10.v13.releasedAt = t0;
            int permits = 0;
            for (double s = 0; s <= 30; s += .1) {
                auto t = t0 + chrono::milliseconds(int(s * 1000));
                e.stackVision.lastFrame = t;
                e.stackVision.clearStable = true;
                e.stackVision.clearFrames = 5;
                size_t before = sent.size();
                v13Observe(e, t);
                v13StackSupervise(e, t);
                for (size_t i = before; i < sent.size(); ++i)
                    if (sent[i] == permit) ++permits;
                if (permits) break;
            }
            if (permits != 1 || !frames236().empty()) fail("Lost stack permit was not resent identically");
            cout
                << "PASS V13.5 lost frame: armless + unchanged scene resends the identical plan/permit (2x) before any re-pick; arm seen -> normal re-pick\n";
        }
        // 8. Pixel-level stack chain with strong perspective (side faces 18-22 px): the square-top detector
        //    alone fails here; the before/after stack comparison must verify, detect misses and wrong layers.
        {
            Scalar inkC[3] = { Scalar(40, 40, 190), Scalar(50, 150, 60), Scalar(190, 90, 40) };
            auto drawStack = [&](Mat& f, const array<int, 3>& col, int layers, int side, Point foot) {
                for (int l = 0; l < layers; ++l) {
                    Point p = foot - Point(0, side * (l + 1));
                    rectangle(f, Rect(p.x, p.y + 34, 34, side), inkC[col[l]] * 0.55, FILLED);
                    rectangle(f, Rect(p.x, p.y, 34, 34), inkC[col[l]], FILLED);
                }
                };
            auto drawBlock = [&](Mat& f, int colour, Point2d world) {
                Point px(cvRound(world.x - 300), cvRound(world.y + 200));
                rectangle(f, Rect(px.x - 17, px.y - 17, 34, 34), inkC[colour], FILLED);
                };
            struct Run {
                int permits[4] = { 0, 0, 0, 0 };
                int repicks = 0;
            };
            // layersShown(k): what is physically on the stack after transfer k; sources(k): blocks still on the table.
            auto runCycle = [&](int side, function<int(int)> layersAfter, function<int(int)> colourAt, bool foreign,
                Run& out, App& a, calibrationui::App& c) {
                    setup(a, c);
                    a.v10.phase = 1;
                    a.v10.handshakeStep = 1;
                    a.v10.pick = { {{420, -100}, {480, 40}, {420, 40}} };
                    Mat bg(480, 640, CV_8UC3, Scalar(150, 150, 150));
                    a.v10.emptyBackground = bg.clone();
                    auto& sv = a.stackVision;
                    sv = StackVision{};
                    sv.configured = true;
                    sv.reference = bg.clone();
                    sv.workspace = Rect(10, 10, 620, 460);
                    sv.stack = Rect(18, 331, 110, 110);
                    sv.faceArea = 34 * 34;
                    sv.begin(a.v10.colours);
                    auto model = stackColourModel(a);
                    Point foot(56, 400);
                    auto t0 = clock::now() + chrono::hours(1);
                    auto t = t0;
                    auto frameFor = [&](int placed, bool arm) {
                        Mat f = bg.clone();
                        array<int, 3> col{ {colourAt(0), colourAt(1), colourAt(2)} };
                        drawStack(f, col, placed, side, foot);
                        for (int l = placed; l < 3; ++l) drawBlock(f, a.v10.colours[l], a.v10.pick[l]);
                        if (foreign && placed >= 2) rectangle(f, Rect(100, 340, 30, 30), inkC[0], FILLED);
                        if (arm) rectangle(f, Rect(30, 120, 260, 300), Scalar(70, 70, 70), FILLED);
                        return f;
                        };
                    auto tick = [&](const Mat& f) {
                        t += chrono::milliseconds(100);
                        c.frame = f;
                        c.detections = calibrationui::detect(f, model);
                        sv.observe(f, c.detections, t);
                        v13Observe(a, t);
                        if (a.v10.phase == 1) v11StackHandshake(a, t);
                        if (a.v10.phase == 1) v13StackSupervise(a, t);
                        };
                    for (int i = 0; i < 10; ++i) tick(frameFor(0, false));
                    v13Release(a, 1, t);
                    int robotAt = 1; // robot executes transfer `robotAt` once released towards it
                    for (int guard = 0; guard < 4000 && a.v10.phase == 1; ++guard) {
                        size_t before = sent.size();
                        int shown = layersAfter(robotAt);
                        // 3 s arm in view, then parked at HOME
                        for (int i = 0; i < 30 && a.v10.phase == 1; ++i) tick(frameFor(min(shown, robotAt - 1), true));
                        for (int i = 0; i < 250 && a.v10.phase == 1; ++i) {
                            tick(frameFor(shown, false));
                            bool moved = false;
                            for (size_t q = before; q < sent.size(); ++q) {
                                auto& fr = sent[q];
                                if (fr.size() < 8) continue;
                                if (fr[1] == 234 && fr[5] == 4 && fr[7] == robotAt) {
                                    ++out.permits[robotAt];
                                    moved = true;
                                }
                                if (fr[1] == 236 && fr[7] == robotAt) {
                                    ++out.repicks;
                                    moved = true;
                                }
                            }
                            if (moved) {
                                bool permit = sent.back()[1] == 234;
                                if (permit) {
                                    if (robotAt == 3) {
                                        for (int j = 0; j < 30; ++j) tick(frameFor(shown, false));
                                        return;
                                    }
                                    ++robotAt;
                                }
                                v13Release(a, robotAt, t);
                                if (!permit) a.v10.v13.pendingAttempt = 0;
                                break;
                            }
                        }
                        if (sent.size() == before) return; // waiting: no command at all
                    }
                };
            for (int side : {18, 22}) {
                sent.clear();
                App a;
                calibrationui::App c;
                Run run;
                runCycle(
                    side, [](int k) { return k; }, [&](int l) { return array<int, 3>{{2, 0, 1}}[l]; }, false, run, a,
                    c);
                if (run.permits[1] < 1 || run.permits[2] < 1 || run.permits[3] < 1 || run.repicks || a.v10.phase == 5)
                    fail("Perspective stack side " + to_string(side) + " px not verified: permits " +
                        to_string(run.permits[1]) + "/" + to_string(run.permits[2]) + "/" + to_string(run.permits[3]) +
                        " status: " + a.v10.status);
                if (a.stackVision.progress != 3 || a.stackVision.observedLayers != a.v10.colours ||
                    !a.v10.v13.deltaOK[3])
                    fail("Stack record does not match the physical stack: progress " +
                        to_string(a.stackVision.progress) + " layers " + to_string(a.stackVision.observedLayers[0]) +
                        to_string(a.stackVision.observedLayers[1]) + to_string(a.stackVision.observedLayers[2]) +
                        " deltaOK " + to_string(a.v10.v13.deltaOK[1]) + to_string(a.v10.v13.deltaOK[2]) +
                        to_string(a.v10.v13.deltaOK[3]));
                if (a.v10.phase != 1 || !v13StackComplete(a, a.stackVision.lastFrame))
                    fail("Verified 3-layer stack did not open the return gate");
                cout
                    << "PASS V13 stack, side faces " << side
                    << " px: every layer verified (by the square face where it is visible, otherwise by the stack change), swatches = physical stack, return gate open\n";
            }
            { // layer 2 not placed on the first attempt (still at its source): re-pick, then verified
                sent.clear();
                App a;
                calibrationui::App c;
                Run run;
                int attempt2 = 0;
                runCycle(
                    20,
                    [&](int k) {
                        if (k == 2 && attempt2++ == 0) return 1;
                        return k;
                    },
                    [&](int l) { return array<int, 3>{{2, 0, 1}}[l]; }, false, run, a, c);
                if (run.repicks != 1 || run.permits[3] < 1 || a.v10.phase == 5)
                    fail("Perspective miss not handled: repicks " + to_string(run.repicks) +
                        " status: " + a.v10.status);
                cout
                    << "PASS V13 stack miss at layer 2 (side faces 20 px): 'nothing changed' -> one re-pick -> layers 2 and 3 verified\n";
            }
            { // wrong colour placed at layer 2: HOLD, never a permit
                sent.clear();
                App a;
                calibrationui::App c;
                Run run;
                runCycle(
                    20, [](int k) { return k; }, [&](int l) { return array<int, 3>{{2, 1, 1}}[l]; }, false, run, a, c);
                if (a.v10.phase != 5 || run.permits[2]) fail("Wrong colour layer not held: " + a.v10.status);
                cout << "PASS V13 wrong colour added at layer 2: HOLD, no permit\n";
            }
            { // separate block lands beside the stack: HOLD
                sent.clear();
                App a;
                calibrationui::App c;
                Run run;
                runCycle(
                    20, [](int k) { return k; }, [&](int l) { return array<int, 3>{{2, 0, 1}}[l]; }, true, run, a, c);
                if (a.v10.phase != 5 || run.permits[2]) fail("Foreign block in stack zone not held: " + a.v10.status);
                cout << "PASS V13 separate block beside the stack: HOLD, no permit\n";
            }
        }
        // 9. Outline guard and centre-line shift are OFF: pick targets must equal the V12 top face.
        {
            calibrationui::H h(1, 0, 300, 0, 1, -200, 0, 0, 1);
            calibrationui::ColourModel cfg;
            cfg.staticStyle = false;
            cfg.sat = 55;
            cfg.val = 35;
            cfg.hue = { 0, 60, 110 };
            cfg.tol = { 13, 25, 20 };
            Mat f(480, 640, CV_8UC3, Scalar(150, 150, 150));
            rectangle(f, Rect(120, 150, 35, 35), Scalar(190, 90, 40), FILLED);
            rectangle(f, Rect(120, 185, 35, 8), Scalar(95, 45, 20), FILLED);
            calibrationui::Detection d;
            d.colour = 2;
            d.quad = { Point(120, 158), Point(155, 158), Point(155, 193), Point(120, 193) };
            d.box = boundingRect(d.quad);
            d.centre = calibrationui::crossCentre(d.quad);
            d.area = 35 * 35;
            d.pickFaceVerified = true;
            auto g = v13ConstrainY(f, cfg, h, d);
            if (norm(g.centre - d.centre) > 1e-9 || isfinite(g.blobMinY)) fail("Outline guard still moves targets");
            if (calibrationui::v13Optics().enabled) fail("Centre-line shift enabled without a measured centre line");
            cout << "PASS V13 pick targets: outline re-centring and centre-line shift are off (V12 top face used)\n";
        }
        // 10. Centre-line (camera nadir) anchoring and lamp-tint-safe outline.
        {
            calibrationui::H h(1, 0, 300, 0, 1, -200, 0, 0, 1);
            calibrationui::ColourModel cfg;
            cfg.staticStyle = false;
            cfg.sat = 55;
            cfg.val = 35;
            cfg.hue = { 0, 60, 110 };
            cfg.tol = { 13, 25, 20 };
            // Block at robot (420,-100); the centre line is at Y 28, so its +Y side faces the camera.
            Mat f(480, 640, CV_8UC3, Scalar(150, 150, 150));
            rectangle(f, Rect(103, 83, 35, 35), Scalar(190, 90, 40), FILLED);
            rectangle(f, Rect(103, 118, 35, 7), Scalar(95, 45, 20), FILLED);
            calibrationui::Detection d;
            d.colour = 2;
            d.quad = { Point(103, 89), Point(138, 89), Point(138, 124), Point(103, 124) };
            d.box = boundingRect(d.quad);
            d.centre = calibrationui::crossCentre(d.quad);
            d.area = 35 * 35;
            d.pickFaceVerified = true;
            double x0, x1, y0, y1;
            if (!calibrationui::v13Outline(f, cfg, h, d, x0, x1, y0, y1)) fail("Outline not measured");
            const auto& o = calibrationui::v13Optics();
            double cy = calibrationui::v13AnchorAxis(y0, y1, o.nadirY, o.sidePerMm),
                cx = calibrationui::v13AnchorAxis(x0, x1, o.nadirX, o.sidePerMm);
            double trueY = 83 + 17.5 - 200, trueX = 103 + 17.5 + 300, midY = (y0 + y1) / 2;
            if (!isfinite(cy) || abs(cy - trueY) > 1.2 || abs(cx - trueX) > 1.2 || abs(midY - trueY) < 2.5)
                fail("Centre-line anchor missed the true top: " + to_string(cy) + " vs " + to_string(trueY));
            // Lamp-tinted pink background (LIGHT OFF) must not be counted as a red block.
            Mat pink(480, 640, CV_8UC3, Scalar(165, 150, 190));
            rectangle(pink, Rect(103, 83, 35, 35), Scalar(40, 40, 190), FILLED);
            calibrationui::Detection r = d;
            r.colour = 0;
            r.quad = { Point(103, 83), Point(138, 83), Point(138, 118), Point(103, 118) };
            r.box = boundingRect(r.quad);
            r.centre = calibrationui::crossCentre(r.quad);
            if (!calibrationui::v13Outline(pink, cfg, h, r, x0, x1, y0, y1) || y1 - y0 > 38 || x1 - x0 > 38)
                fail("Tinted background leaked into the red outline");
            cout << "PASS V13 centre line: 7 mm side towards the camera centre -> centre "
                << calibrationui::fmt(abs(cy - trueY), 1) << " mm from the true top (outline midpoint would be "
                << calibrationui::fmt(abs(midY - trueY), 1) << " mm off); pink LIGHT OFF tint not counted as red\n";
        }
        if (!onPath.empty()) { // the two supplied screenshots through the real pickup path
            vector<Mat> im;
            for (auto pth : { onPath, offPath }) {
                Mat m = imread(pth);
                if (m.cols == 639 && m.rows >= 507) m = m(Rect(0, 30, 639, 477)).clone();
                im.push_back(m);
            }
            calibrationui::App c;
            c.opt.source = "image";
            c.frame = im[0];
            for (int i = 0; i < 25; ++i) c.update();
            if (c.grid.size() == 9 && calibrationui::makeMapping(c.grid, c.refs, c.frame.size(), c.mapping)) {
                auto cfg = c.cfg;
                cfg.staticStyle = true;
                cfg.sat = 65;
                cfg.val = 60;
                array<vector<pair<int, Point2d>>, 2> v12, v13;
                int refusedOff = 0;
                for (int k = 0; k < 2; ++k)
                    for (auto& det : calibrationui::detect(im[k], cfg)) {
                        auto a0 = calibrationui::v11TopFace(im[k], cfg, c.mapping.h, det);
                        auto a1 = v13ConstrainY(im[k], cfg, c.mapping.h,
                            calibrationui::v13PickFace(im[k], cfg, c.mapping.h, det));
                        if (a0.pickFaceVerified)
                            v12[k].push_back({ det.colour, calibrationui::project(c.mapping.h, a0.centre) });
                        if (a1.pickFaceVerified)
                            v13[k].push_back({ det.colour, calibrationui::project(c.mapping.h, a1.centre) });
                        else if (k)
                            ++refusedOff;
                    }
                auto worst = [](array<vector<pair<int, Point2d>>, 2>& r) {
                    double m = 0;
                    for (auto& x : r[0]) {
                        double b = 1e9;
                        for (auto& y : r[1])
                            if (y.first == x.first) b = min(b, norm(y.second - x.second));
                        if (b < 30) m = max(m, b);
                    }
                    return m;
                    };
                double w12 = worst(v12), w13 = worst(v13);
                cout << "LIGHT ON vs OFF, same blocks: worst centre difference V12 " << calibrationui::fmt(w12, 2)
                    << " mm, V13 " << calibrationui::fmt(w13, 2)
                    << " mm; LIGHT OFF blocks refused by the Y guard: " << refusedOff << "\n";
                if (refusedOff) fail("LIGHT OFF tint still makes the guard refuse blocks");
            }
        }
        // 11. Permit lock (race seen live: red returned to the planned spot, PC had retargeted on a resend)
        //     and the outside-box offset allowance.
        {
            sent.clear();
            App a;
            calibrationui::App c;
            setup(a, c);
            a.v10.phase = 3;
            a.v10.handshakeStep = 4;
            a.v10.drop = { {{350, -220}, {520, -40}, {470, -150}} };
            auto& sv = a.stackVision;
            sv.progress = 3;
            sv.clearStable = true;
            sv.clearFrames = 5;
            c.detections = { block(1, {470, -150}) };
            a.v10.currentReturns[2] = true;
            auto t0 = clock::now();
            for (int i = 0; i < 6; ++i) a.v10.returnEvidence[2].update(true, t0 + chrono::milliseconds(i * 100));
            auto tick = [&](double s) {
                auto t = t0 + chrono::milliseconds(int(s * 1000));
                sv.lastFrame = t;
                a.v10.controlLast = {};
                int returnIndex = 2;
                if (v13ReturnDone(a, returnIndex, t)) {
                    Point2d next;
                    bool fresh = false;
                    int mode = v13NextDropDecision(a, 4, next, fresh);
                    if (mode == 2) {
                        v13SendTarget(a, 2, 4, a.v10.v13.retargetAttempt[4], next, fresh, t);
                        a.v10.v13.permitKind[4] = 2;
                    }
                    else if (mode == 0) {
                        v11SendControl(a, 4, 4);
                        a.v10.v13.permitKind[4] = 1;
                    }
                }
                };
            tick(.5);
            if (sent.empty() || sent.back()[1] != 234) fail("Free next drop did not get a plain permit");
            Point2d planned = a.v10.drop[1];
            c.detections.push_back(block(0, planned)); // arm/other block now covers the planned drop
            tick(3.1);
            tick(5.7);
            for (auto& q : sent)
                if (q[1] == 236) fail("Retargeted a step whose plain permit had already been sent");
            if (norm(a.v10.drop[1] - planned) > 1e-9) fail("PC changed the planned drop after the robot was released");
            calibrationui::App c2;
            App b;
            b.calibration = &c2;
            c2.mapping.robot = calibrationui::nominalRobot();
            v13Allowance().yMinus = .03;
            v13Allowance().cap = 8;
            auto inBox = v13AllowanceAt(b, { 442, -8 }), left = v13AllowanceAt(b, { 442, -182.8 }),
                far = v13AllowanceAt(b, { 442, -600 });
            if (norm(inBox) > 1e-9 || abs(left.y + 3.0) > 1e-6 || left.x != 0 || abs(far.y + 8) > 1e-6)
                fail("Offset allowance wrong: inside " + to_string(inBox.y) + " left " + to_string(left.y));
            v13Allowance().yMinus = 0;
            cout
                << "PASS V13 permit lock: plain permit stays plain on resends (no retarget after release); allowance 0 inside P1-P9, -3.0 mm at 100 mm left of the box, capped at -8 mm\n";
        }
        // 12. TEST page AL/CL/CR buttons and the centre-line Y (all colours).
        {
            App a;
            calibrationui::App c;
            a.calibration = &c;
            a.calibrationMode = true;
            a.dynamicTestMode = true;
            c.opt.source = "demo";
            for (int i = 0; i < 25; ++i) {
                c.frame = calibrationui::synthetic(c);
                c.update();
            }
            if (!calibrationui::makeMapping(c.grid, c.refs, c.frame.size(), c.mapping)) fail("demo mapping");
            c.valid = true;
            c.test = true;
            c.mapping.robot = calibrationui::nominalRobot();
            auto& g = v13Allowance();
            g.blueY = .03;
            g.yMinus = .03;
            auto keep = g;
            auto click = [&](int x, int y) { mouse(EVENT_LBUTTONDOWN, x, y, 0, &a); };
            g.centreYPlus = 0;
            keep = g;
            click(960, 790);
            if (abs(g.yMinus - (keep.yMinus + .005)) > 1e-9) fail("AL + did not raise the left allowance");
            // CL/CR act on the SELECTED block's colour only.
            if (c.detections.empty()) fail("demo has no detections");
            c.selectionMissing = false;
            c.selected = 0;
            int sel = c.detections[0].colour, other = (sel + 1) % 3;
            click(1095, 790);
            if (abs(g.colourYMinus[sel] - (keep.blueY + .005)) > 1e-9) fail("CL + did not raise the selected colour");
            if (abs(g.yMinusFor(other) - keep.blueY) > 1e-9 || abs(g.blueY - keep.blueY) > 1e-9)
                fail("CL + changed another colour");
            click(1230, 790);
            click(1230, 790);
            if (abs(g.colourYPlus[sel] - .01) > 1e-9 || abs(g.yPlusFor(other)) > 1e-9) fail("CR + per colour wrong");
            click(1030, 790);
            click(895, 790);
            if (abs(g.colourYMinus[sel] - keep.blueY) > 1e-9 || abs(g.yMinus - keep.yMinus) > 1e-9) fail("AL -/CL - did not lower");
            // Per-colour targets: lower gain for the selected colour only.
            g.colourYMinus[sel] = .01;
            Point2d leftP(442, -107.8);
            if (abs(v13Target(a, leftP, sel).y - (leftP.y - 0 * 0 - .01 * 100 - .03 * 25)) > 1e-6 ||
                abs(v13Target(a, leftP, other).y - (leftP.y - .03 * 100 - .03 * 25)) > 1e-6)
                fail("Per-colour centre gain not applied");
            c.selectionMissing = true;
            double before = g.colourYMinus[sel];
            click(1095, 790);
            if (g.colourYMinus[sel] != before) fail("CL changed a gain with no block selected");
            g.colourYMinus = { {NAN, NAN, NAN} };
            g.colourYPlus = { {NAN, NAN, NAN} };
            g.centreYPlus = .01; // shared gain for the all-colour checks below
            click(470, 790);
            if (abs(a.dynamicTrim.x + .5) > 1e-9) fail("Trim buttons broken");
            a.dynamicTrim = {};
            g.yMinus = 0;
            double yc = -7.8;
            Point2d left(442, yc - 100), right(442, yc + 50), farRight(442, yc + 1000);
            for (int colour = 0; colour < 3; ++colour) {
                auto tl = v13Target(a, left, colour), tR = v13Target(a, right, colour),
                    tF = v13Target(a, farRight, colour);
                if (abs(tl.y - (left.y - g.blueY * 100)) > 1e-6) fail("Centre Y minus wrong for colour " + to_string(colour));
                if (abs(tR.y - (right.y + .01 * 50)) > 1e-6) fail("Centre Y plus wrong for colour " + to_string(colour));
                if (abs(tF.y - (farRight.y + g.cap)) > 1e-6) fail("Centre Y plus not capped");
                if (abs(tl.x - left.x) > 1e-9 || abs(tR.x - right.x) > 1e-9) fail("Centre line moved X");
            }
            g.centreYPlus = 0;
            if (abs(v13Target(a, right, 1).y - right.y) > 1e-9) fail("Centre Y plus 0 must leave right side unchanged");
            g = restore.keep;
            v13SaveAllowance();
            g.blueY = 0;
            g.yMinus = 0;
            g.centreYPlus = 0;
            g.colourYMinus = { {NAN, NAN, NAN} };
            g.colourYPlus = { {NAN, NAN, NAN} };
            cout << "PASS V13 centre line: CL/CR change only the selected colour (refused with none selected); RED, GREEN and BLUE 100 mm left get Y - "
                << calibrationui::fmt(keep.blueY * 100, 1)
                << " mm, 50 mm right get Y + 0.5 at gain 0.010 (capped), none when Y+ is 0; AL/CL/CR save; trim unaffected\n";
        }
        // 13. Ambiguous top face: calibration refuses, pickup uses the halfway point (no flip-flop).
        {
            calibrationui::H h(1, 0, 300, 0, 1, -200, 0, 0, 1);
            calibrationui::ColourModel cfg;
            cfg.staticStyle = true;
            cfg.sat = 65;
            cfg.val = 60;
            cfg.hue = { 0, 60, 110 };
            cfg.tol = { 12, 15, 15 };
            cfg.minArea = int(MIN_AREA);
            Mat f(480, 640, CV_8UC3, Scalar(150, 150, 150));
            rectangle(f, Rect(120, 150, 35, 35), Scalar(190, 90, 40), FILLED);
            rectangle(f, Rect(120, 185, 35, 9), Scalar(120, 60, 28), FILLED);
            auto det = calibrationui::detect(f, cfg);
            if (det.empty()) fail("synthetic blue not detected");
            auto strict = calibrationui::v11TopFace(f, cfg, h, det[0]),
                pick1 = calibrationui::v11TopFace(f, cfg, h, det[0], true),
                pick2 = calibrationui::v11TopFace(f, cfg, h, det[0], true);
            if (!pick1.pickFaceVerified || norm(pick1.centre - pick2.centre) > 1e-9) fail("Pickup face not stable");
            cout << "PASS V13 ambiguous face: calibration "
                << (strict.pickFaceVerified ? "accepts (not ambiguous here)" : "refuses")
                << ", pickup returns one stable face\n";
        }
        // 14. Deliberate offsets are not counted against the gripper opening (live: "Gripper opening cannot clear the BLUE face").
        {
            App a;
            calibrationui::App c;
            a.calibration = &c;
            c.valid = true;
            c.mapping.h = calibrationui::H(1, 0, 300, 0, 1, -200, 0, 0, 1);
            c.mapping.robot = calibrationui::nominalRobot();
            v13Allowance().yMinus = .03;
            v13Allowance().blueY = .03;
            v13Allowance().cap = 8;
            calibrationui::Detection d;
            d.colour = 2;
            Point2d px(142, 30); // robot (442,-170): 87 mm outside the box, 162 mm left of centre
            d.quad = { Point(125, 13), Point(160, 13), Point(160, 48), Point(125, 48) };
            d.box = boundingRect(d.quad);
            d.centre = calibrationui::crossCentre(d.quad);
            d.pickFaceVerified = true;
            auto raw = calibrationui::project(c.mapping.h, d.centre);
            auto target = v13Target(a, raw, 2);
            auto corr = v13Correction(a, raw, 2);
            if (abs(target.y - raw.y) < 5) fail("Test needs a large deliberate offset");
            if (v11GripperOpeningFits(d, c.mapping.h, target)) fail("Test premise: uncorrected check should reject");
            if (!v11GripperOpeningFits(d, c.mapping.h, target, corr))
                fail("Deliberate offset still rejected by the opening check");
            if (v11GripperOpeningFits(d, c.mapping.h, target + Point2d(0, 6), corr))
                fail("Real 6 mm aiming error no longer rejected");
            v13Allowance().yMinus = 0;
            v13Allowance().blueY = 0;
            cout << "PASS V13 opening check: BLUE with " << calibrationui::fmt(raw.y - target.y, 1)
                << " mm deliberate Y minus is pickable; a real 6 mm aiming error is still rejected\n";
        }
        // 7. Lighting: global exposure compensation on the supplied lamp ON / OFF images.
        if (!onPath.empty()) {
            Mat on = imread(onPath), off = imread(offPath);
            if (on.empty() || off.empty()) fail("Cannot read lighting images");
            auto crop = [](Mat m) {
                if (m.cols == 639 && m.rows >= 507) m = m(Rect(0, 30, 639, 477)).clone();
                return m;
                };
            on = crop(on);
            off = crop(off);
            Scalar shift = v13ExposureShift(on, off);
            int raw = 0, fixed = 0, total = 0;
            for (int y = 0; y + 40 <= on.rows; y += 40)
                for (int x = 0; x + 40 <= on.cols; x += 40) {
                    Rect r(x, y, 40, 40);
                    ++total;
                    Mat delta, gray, mask;
                    absdiff(on(r), off(r), delta);
                    cvtColor(delta, gray, COLOR_BGR2GRAY);
                    threshold(gray, mask, 30, 255, THRESH_BINARY);
                    if (countNonZero(mask) < .06 * mask.total()) ++raw;
                    v13CompensatedDelta(on(r), off(r), shift, delta);
                    cvtColor(delta, gray, COLOR_BGR2GRAY);
                    threshold(gray, mask, 30, 255, THRESH_BINARY);
                    if (countNonZero(mask) < .06 * mask.total()) ++fixed;
                }
            cout << "LIGHTING ON->OFF neutral shift B" << shift[0] << " G" << shift[1] << " R" << shift[2]
                << "; unchanged 40x40 patches read as unchanged: raw " << raw << "/" << total << ", compensated "
                << fixed << "/" << total << "\n";
            if (fixed < raw) fail("Exposure compensation made the empty test worse");
            Mat moved = off.clone();
            rectangle(moved, Rect(500, 380, 40, 40), Scalar(40, 40, 200), FILLED);
            Mat delta, gray, mask;
            v13CompensatedDelta(on(Rect(500, 380, 40, 40)), moved(Rect(500, 380, 40, 40)), shift, delta);
            cvtColor(delta, gray, COLOR_BGR2GRAY);
            threshold(gray, mask, 30, 255, THRESH_BINARY);
            if (countNonZero(mask) < .06 * mask.total()) fail("Compensation hid a real block");
            cout
                << "PASS V13 lighting: compensation never hides a real block and does not reduce correct empty readings\n";
        }
        return 0;
    }
    catch (const exception& e) {
        serialTestCapture = nullptr;
        cerr << "V13 SELF-TEST FAILED: " << e.what() << endl;
        return 1;
    }
}

// ------------------------------------------------------------
// SECTION 14  PROGRAM ENTRY AND MAIN LOOP
// ------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc >= 2 && string(argv[1]) == "--v13-self-test")
        return v13SelfTest(argc >= 4 ? argv[2] : "", argc >= 4 ? argv[3] : "");
    if (argc >= 2 && string(argv[1]) == "--smooth-cycle-self-test") return smoothCycleSelfTest();
    if (argc >= 4 && string(argv[1]) == "--scene-pair-self-test") return scenePairSelfTest(argv[2], argv[3]);
    if (argc >= 2 && string(argv[1]) == "--lighting-reference-self-test") return lightingReferenceSelfTest();
    if (argc >= 2 && string(argv[1]) == "--lighting-setup-preview") return lightingSetupPreview();
    if (argc >= 2 && string(argv[1]) == "--v11-recovery-self-test") return v11RecoverySelfTest();
    if (argc >= 2 && string(argv[1]) == "--responsive-vision-self-test") return responsiveVisionSelfTest();
    if (argc >= 2 && string(argv[1]) == "--fit-review-preview") return fitReviewPreview();
    if (argc >= 2 && string(argv[1]) == "--ui-audit-self-test") return uiAuditSelfTest();
    if (argc >= 2 && string(argv[1]) == "--saved-reference-self-test") return savedReferenceSelfTest();
    if (argc >= 2 && string(argv[1]) == "--enlarged-hover-preview") return enlargedHoverPreview();
    if (argc >= 3 && string(argv[1]) == "--selected-capture-test") return selectedCaptureTest(argv[2]);
    if (argc >= 3 && string(argv[1]) == "--detector-capture-test")
        return detectorCaptureTest(argv[2], argc >= 4 && string(argv[3]) == "shared");
    if (argc >= 2 && string(argv[1]) == "--v10-preview") return v10Preview();
    if (argc >= 2 && string(argv[1]) == "--calibration-self-test") {
        try {
            return calibrationui::selfTest() ? 0 : 1;
        }
        catch (const exception& e) {
            cerr << e.what() << endl;
            return 1;
        }
    }
    if (argc >= 2 && string(argv[1]) == "--v10-self-test") return v10SelfTest();
    if (argc >= 2 && string(argv[1]) == "--self-test") return measurementSelfTest();
    cout << BUILD_TAG << endl;
    App a;
    loadConfig(a);
    if (fileutil::exists("v10_workspace.yml")) {
        FileStorage f("v10_workspace.yml", FileStorage::READ);
        f["min_x_mm"] >> a.v10.minX;
        f["max_x_mm"] >> a.v10.maxX;
        f["min_y_mm"] >> a.v10.minY;
        f["max_y_mm"] >> a.v10.maxY;
        if (!isfinite(a.v10.minX) || !isfinite(a.v10.maxX) || !isfinite(a.v10.minY) || !isfinite(a.v10.maxY) ||
            a.v10.minX >= a.v10.maxX || a.v10.minY >= a.v10.maxY) {
            cerr << "Invalid v10_workspace.yml" << endl;
            return 1;
        }
    }
    else {
        FileStorage f("v10_workspace.yml", FileStorage::WRITE);
        f << "min_x_mm" << a.v10.minX << "max_x_mm" << a.v10.maxX << "min_y_mm" << a.v10.minY << "max_y_mm"
            << a.v10.maxY;
    }
    if (fileutil::exists("v10_hover_workspace.yml")) {
        FileStorage f("v10_hover_workspace.yml", FileStorage::READ);
        if (!f["min_x_centre_mm"].empty()) f["min_x_centre_mm"] >> a.hoverMinX;
        if (!f["max_x_centre_mm"].empty()) f["max_x_centre_mm"] >> a.hoverMaxX;
        if (!f["min_y_centre_mm"].empty()) f["min_y_centre_mm"] >> a.hoverMinY;
        if (!f["max_y_centre_mm"].empty()) f["max_y_centre_mm"] >> a.hoverMaxY;
        if (!isfinite(a.hoverMinX) || !isfinite(a.hoverMaxX) || !isfinite(a.hoverMinY) || !isfinite(a.hoverMaxY) ||
            a.hoverMinX >= a.hoverMaxX || a.hoverMinY >= a.hoverMaxY || abs(a.hoverMinX) > 1000 ||
            abs(a.hoverMaxX) > 1000 || abs(a.hoverMinY) > 1000 || abs(a.hoverMaxY) > 1000) {
            cerr << "Invalid v10_hover_workspace.yml" << endl;
            return 1;
        }
    }
    else {
        FileStorage f("v10_hover_workspace.yml", FileStorage::WRITE);
        f << "min_x_centre_mm" << a.hoverMinX << "max_x_centre_mm" << a.hoverMaxX << "min_y_centre_mm" << a.hoverMinY
            << "max_y_centre_mm" << a.hoverMaxY;
    }
    if (fileutil::exists("v10_cycle_state.yml")) {
        FileStorage f("v10_cycle_state.yml", FileStorage::READ);
        int active = 0;
        f["active"] >> active;
        f["token"] >> a.v10.token;
        if (active) {
            a.v10.phase = 5;
            a.v10.status =
                "Previous dynamic cycle unresolved. Restore physical/controller state before UI recovery reset.";
        }
    }

    calibrationui::App calibration;
    calibration.opt.source = "camera";
    calibration.opt.camera = CAM;
    calibration.buttons.back().label = "Q Back";
    a.calibration = &calibration;
    if (fileutil::exists("calibration_reference_points.csv") &&
        !calibrationui::loadReferences("calibration_reference_points.csv", calibration.refs)) {
        cerr << "Invalid calibration_reference_points.csv" << endl;
        return 1;
    }
    bool previewCalibration = argc >= 2 && string(argv[1]) == "--preview-calibration";
    if (!previewCalibration && calibration.loadPersistentCalibration())
        cout << "Loaded saved calibration (calibration_saved.yml): TEST is ready, no recalibration needed.\n";
    loadLightingPreference(a);
    loadDetectionTuning();
    if (!previewCalibration) loadStackView(a);
    bool preview =
        previewCalibration || (argc >= 2 && (string(argv[1]) == "--preview" || string(argv[1]) == "--preview-setup"));
    if (preview) {
        a = App{};
        a.setupOpen = string(argv[1]) == "--preview-setup";
        a.autoProposal = false;
    }
    a.calibration = &calibration;
    if (previewCalibration) {
        a.calibrationMode = true;
        calibration.opt.source = "demo";
    }
    if (argc >= 2 && !preview)
        startSerial(a, argv[1]);
    else
        cout << "Usage: HandEye_COM_Direct.exe COM3\n";

    VideoCapture cap;
    if (!preview) cap.open(CAM);
    if (!preview && !cap.isOpened()) {
        cerr << "Cannot open camera index " << CAM << "\n";
        disconnect(a);
        return 1;
    }
    namedWindow("Hand Eye Robot Console", WINDOW_NORMAL);
    resizeWindow("Hand Eye Robot Console", W, H);
    setMouseCallback("Hand Eye Robot Console", mouse, &a);
    cout << "CYCLE: 1/2/3 select -> S send stack -> automatic home / random return / release.\n"
        << "Map once with M (or auto-propose with nine blocks); V verifies.\n"
        << "R/G/B sample colour then click block, K reset HSV, P refresh plans, Q quit.\n"
        << "Z opens Calibration setup; X clears stack observation history.\n";

    auto lastPump = chrono::steady_clock::now();
    serialVisionPump = [&]() { // Keep checking camera images during serial transmission.
        if (a.v10.pumpFailed) return;
        try {
            auto now = chrono::steady_clock::now();
            if (preview || chrono::duration<double>(now - lastPump).count() < .08) return;
            lastPump = now;
            Mat live;
            if (!cap.read(live) || live.empty()) return;
            a.lastFrame = live.clone();
            a.cameraSize = live.size();
            if (a.dynamicOperate && !calibration.freeze) {
                calibration.frame = live;
                calibration.update();
                v11RefinePickupFaces(a);
                updateStackVision(a, calibration.frame);
                // Fresh evidence keeps accumulating while the independent sender runs.
                // This pump never transmits, advances a checkpoint, or clears recovery.
                auto evidenceNow = chrono::steady_clock::now();
                if (a.v10.phase == 1 || a.v10.phase == 2)
                    for (int i = 0; i < 3; ++i)
                        a.v10.sourceEvidence[i].update(v12PickupVerified(a, i, evidenceNow), evidenceNow);
                if (a.v10.phase == 3 || a.v10.phase == 4) v10ObserveReturns(a, evidenceNow);
                drawDynamic(calibration.ui, a);
                drawNavigation(calibration.ui, a);
                label(calibration.ui, "Sending to the robot...", { 22, 875 }, .34, GOLD);
                imshow("Hand Eye Robot Console", calibration.ui);
            }
            else {
                Mat tx(900, 1280, CV_8UC3, BG);
                resize(live, tx(Rect(15, 100, 820, 615)), { 820, 615 });
                drawNavigation(tx, a);
                label(tx, "Sending to the robot...", { 22, 760 }, .5, GOLD);
                imshow("Hand Eye Robot Console", tx);
            }
        }
        catch (const exception& e) {
            a.v10.pumpFailed = true;
            a.v10.startRequested = false;
            a.v10.phase = 5;
            a.v10.operatorRequest = a.v10.retryControl = 1;
            a.message = a.dynamicStatus = a.v10.status =
                string("Vision failure during serial TX: ") + e.what() + ". Further transfers locked; inspect pendant.";
        }
        };
    bool enlargedLoaded = false;
    while (true) {
        if (a.pendingControllerMode && a.serialConnected && !a.busy && !a.sentCount && !a.dynamicStep &&
            !a.buildCount && !a.v10.phase && !calibration.entering && !calibration.diagnosticRemaining) {
            int requested = a.pendingControllerMode;
            a.pendingControllerMode = 0;
            sendControllerMode(a, requested);
        }
        Mat frame;
        if (preview) {
            frame = Mat(620, 850, CV_8UC3, Scalar(34, 39, 45));
            for (int i = 0; i < 9; ++i) {
                Point centre(180 + (i % 3) * 180, 150 + (i / 3) * 160);
                rectangle(frame, Rect(centre.x - 30, centre.y - 30, 60, 60), ink(i % 3 + 1), FILLED);
                a.points[i] = centre;
            }
            a.mapped = N;
            a.verifiedMap = true;
        }
        else
            cap >> frame;
        if (frame.empty()) break;
        a.lastFrame = frame.clone();
        a.cameraSize = frame.size();
        if (!preview) v11ServiceOperator(a);
        if (!enlargedLoaded && !preview) {
            calibration.frame = frame.clone();
            try {
                calibration.loadEnlarged();
            }
            catch (const exception& e) {
                calibration.message = e.what();
            }
            enlargedLoaded = true;
        }
        if (a.calibrationMode) {
            if (calibration.staticColourEdited) {
                for (int k = 0; k < 3; ++k) {
                    a.hue[k + 1] = calibration.staticCfg.hue[k];
                    a.tolerance[k + 1] = calibration.staticCfg.tol[k];
                }
                a.satMin = calibration.staticCfg.sat;
                a.valMin = calibration.staticCfg.val;
                saveConfig(a);
                calibration.staticColourEdited = false;
            }
            calibration.enlargedHoverView = a.dynamicTestMode && a.enlargedHover;
            calibration.useStaticDetector = a.dynamicOperate || a.dynamicTestMode;
            calibration.staticCfg.staticStyle = true;
            calibration.staticCfg.sat = a.satMin;
            calibration.staticCfg.val = a.valMin;
            calibration.staticCfg.minArea = int(MIN_AREA);
            for (int k = 0; k < 3; ++k) {
                calibration.staticCfg.hue[k] = a.hue[k + 1];
                calibration.staticCfg.tol[k] = a.tolerance[k + 1];
            }
            if (previewCalibration) {
                for (int i = 0; i < 25; ++i) {
                    calibration.frame = calibrationui::synthetic(calibration);
                    calibration.update();
                }
            }
            if (!calibration.freeze && !previewCalibration) {
                calibration.frame = frame.clone();
                calibration.update();
            }
            try {
                tickLightingCapture(a);
            }
            catch (const exception& e) {
                cancelLightingCapture(a, string("Lighting capture failed: ") + e.what());
            }
            if (a.dynamicOperate || a.dynamicTestMode) v11RefinePickupFaces(a);
            if (a.dynamicOperate) updateStackVision(a, calibration.frame);
            if (a.dynamicRequest) {
                int request = a.dynamicRequest;
                a.dynamicRequest = 0;
                dynamicAction(a, request);
            }
            if (a.dynamicOperate) {
                try {
                    v10Tick(a);
                }
                catch (const exception& e) {
                    a.v10.phase = 5;
                    a.v10.status = e.what();
                }
                drawDynamic(calibration.ui, a);
            }
            else
                calibration.draw();
            drawNavigation(calibration.ui, a);
            if (a.lightingSetupOpen) drawLightingSetup(calibration.ui, a);
            if (a.dynamicTestMode) drawTestControls(a);
            if (!a.dynamicOperate)
                label(calibration.ui,
                    a.dynamicTestMode
                    ? "TEST: fixed Z200.81 / approach286.03 mm. No ACK. Check pendant before each next action."
                    : "O: BACK TO OPERATION | Calibration/test only - no robot commands",
                    Point(20, a.dynamicTestMode ? 893 : 880), .43, GOLD);
            if (previewCalibration) {
                imwrite("operator_calibration_preview.png", calibration.ui);
                break;
            }
            label(calibration.ui, BUILD_TAG, Point(900, 893), .38, MUTED);
            imshow("Hand Eye Robot Console", calibration.ui);
            int ck = waitKey(a.dynamicOperate ? 1 : 25);
            if (a.dynamicOperate) {
                if (ck == 27 || ck == 'q') switchPage(a, 0);
                continue;
            }
            if (a.dynamicStep) {
                if (ck >= 0)
                    a.dynamicStatus = "Session active: use staged test buttons; physical controller STOP for faults";
                continue;
            }
            if (!calibration.entering && ck >= '1' && ck <= '3') {
                switchPage(a, ck - '1');
                continue;
            }
            if (calibration.exit ||
                (!calibration.entering && (ck == 'o' || ck == 'O' || ck == 'q' || ck == 'Q' || ck == 27))) {
                switchPage(a, 0);
            }
            else {
                try {
                    calibration.key(ck);
                    if (a.dynamicTestMode && ck >= 0) a.dynamicStatus = calibration.message;
                }
                catch (const exception& e) {
                    calibration.message = e.what();
                }
            }
            continue;
        }
        Mat smoothed, hsv;
        GaussianBlur(frame, smoothed, Size(5, 5), 0);
        hsv = calibrationui::lightingMonitoredHSV(smoothed, &calibration.lighting);
        a.lastHSV = hsv;
        Mat camera = frame.clone();
        vector<Detection> det;
        for (int c = 1; c <= 3; ++c) findColour(hsv, camera, c, a, det);
        a.lastDetections = det;
        a.lastFrame = frame;
        updateStackVision(a, frame);
        updateBlockCapture(a, det);

        if (a.autoProposal && a.mapped != N && !a.mapping && !a.planLocked && !a.learnPick && !a.learning) {
            array<Point, N> proposed{};
            if (proposeGrid(det, proposed)) {
                bool consistent = a.autoCandidate;
                if (consistent)
                    for (int i = 0; i < N; ++i)
                        if (norm(proposed[i] - a.candidatePoints[i]) > 15) consistent = false;
                a.candidateStreak = consistent ? a.candidateStreak + 1 : 1;
                a.candidatePoints = proposed;
                a.autoCandidate = true;
                if (a.candidateStreak >= STABLE) {
                    a.points = proposed;
                    a.mapped = N;
                    a.verifiedMap = false;
                    resetObservations(a);
                    a.message = "AUTO GRID PROPOSAL: verify P1-P9 labels, press V to accept";
                }
            }
            else {
                a.autoCandidate = false;
                a.candidateStreak = 0;
            }
        }
        if (a.mapped == N && !a.mapping) {
            auto matched = matchDetections(a, det);
            for (int i = 0; i < N; ++i) {
                int observation = matched[i] >= 0 ? det[matched[i]].colour : UNKNOWN;
                auto& s = a.positions[i];
                if (observation == UNKNOWN) {
                    s.candidate = UNKNOWN;
                    s.streak = 0;
                    if (s.missing < LOST_FRAMES) s.missing++;
                    s.uncertain = s.missing < LOST_FRAMES;
                    if (s.missing >= LOST_FRAMES) s.confirmed = UNKNOWN;
                }
                else {
                    s.missing = 0;
                    if (s.candidate == observation)
                        s.streak++;
                    else {
                        s.candidate = observation;
                        s.streak = 1;
                    }
                    s.uncertain = (s.streak < STABLE);
                    if (s.streak >= STABLE) s.confirmed = observation;
                }
            }
        }
        if (a.mapped == N && a.verifiedMap &&
            (!a.planRefreshReady ||
                chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - a.lastPlanRefresh).count() >= 5)) {
            refreshPlans(a);
        }
        for (int j = 0; j < 3; ++j)
            if (a.zoneDefined[j] && !a.stackVision.configured) {
                int seen = UNKNOWN;
                double best = 0;
                for (const auto& d : det) {
                    double distance = norm(d.centre - a.stackZones[j]);
                    if (distance < 55 && d.area > best) {
                        best = d.area;
                        seen = d.colour;
                    }
                }
                if (a.topCandidate[j] == seen)
                    a.topStreak[j]++;
                else {
                    a.topCandidate[j] = seen;
                    a.topStreak[j] = 1;
                }
                if (a.topStreak[j] >= STABLE) {
                    if (seen != a.stackTop[j]) {
                        if (seen != UNKNOWN && (a.observedHistory[j].empty() || a.observedHistory[j].back() != seen))
                            a.observedHistory[j].push_back(seen);
                        a.stackTop[j] = seen;
                    }
                }
                drawMarker(camera, a.stackZones[j], GOLD, MARKER_CROSS, 12, 1);
                label(camera, "STACK S" + to_string(j + 1) + " TOP:" + cname(a.stackTop[j]),
                    a.stackZones[j] + Point(-52, -62), .43, Scalar(230, 205, 75), 2);
            }
        drawStackVision(camera, a);
        // Fresh live observations, independent of the 5-second planning refresh.
        if (a.autoMode && !a.busy) {
            const auto now = chrono::steady_clock::now();
            const chrono::steady_clock::time_point unset{};
            for (int j = 0; j < 3; ++j) {
                if (!a.sentPlans[j]) continue;
                const int saved = a.selected;
                a.selected = j + 1;
                if (!a.returnSent[j]) {
                    a.message = "STACK CHECK: " + a.stackVision.note;
                    int empty = 0;
                    bool sourcesEmpty = true;
                    for (const auto& p : a.positions)
                        if (!p.uncertain && p.confirmed == UNKNOWN) ++empty;
                    for (int src : a.sentSources[j])
                        if (src < 0 || a.positions[src].uncertain || a.positions[src].confirmed != UNKNOWN)
                            sourcesEmpty = false;
                    if (sourcesEmpty && empty >= 3 && a.stackVision.settledReturnReady(now)) {
                        if (a.emptySince[j] == unset) a.emptySince[j] = now;
                        if (chrono::duration<double>(now - a.emptySince[j]).count() >= 3.0) operatorReturn(a);
                    }
                    else
                        a.emptySince[j] = unset;
                }
                else {
                    bool allBack = true;
                    for (int l = 0; l < 3; ++l) {
                        const auto& p = a.positions[a.returnDest[j][l]];
                        if (!p.uncertain && p.confirmed != UNKNOWN) a.returnStarted[j] = true;
                        if (p.uncertain || p.confirmed != a.sentColours[j][l]) allBack = false;
                    }
                    if (allBack) {
                        if (a.restoredSince[j] == unset) a.restoredSince[j] = now;
                        // Returned colours plus fresh empty-stack/workspace-clear evidence.
                        // This is not a joint-pose HOME acknowledgement.
                        double finishElapsed = chrono::duration<double>(now - a.restoredSince[j]).count();
                        a.message = "Returned blocks visible; waiting for empty stack and workspace-clear view. " +
                            a.stackVision.note;
                        if (finishElapsed >= 1.0 && a.stackVision.finishReady(now)) {
                            operatorClear(a);
                            operatorRelease(a);
                        }
                    }
                    else
                        a.restoredSince[j] = unset;
                    // No home ACK exists. Retry IDENTICAL destinations until return begins.
                    if (a.sentPlans[j] && !a.returnStarted[j] && staticReturnClear(a, j) &&
                        a.stackVision.settledReturnReady(now) &&
                        chrono::duration<double>(now - a.retryAt[j]).count() >= 15.0) {
                        const auto& d = a.returnDest[j];
                        const vector<uint8_t> fields = {
                            static_cast<uint8_t>(0x80 + j + 1), static_cast<uint8_t>(0x10 + d[2] + 1),
                            static_cast<uint8_t>(0x20 + d[1] + 1), static_cast<uint8_t>(0x30 + d[0] + 1) };
                        sendFrame(a, 250, fields, 251);
                        a.retryAt[j] = chrono::steady_clock::now();
                    }
                }
                a.selected = saved;
            }
        }
        for (int i = 0; i < a.mapped; ++i) {
            Point p = a.points[i];
            const Position& st = a.positions[i];
            string state = st.uncertain ? "UNCERTAIN" : st.confirmed == UNKNOWN ? "EMPTY" : cname(st.confirmed);
            Scalar c = st.uncertain ? GOLD : ink(st.confirmed);
            // Fixed anchors are small markers; actual block outlines come from contours.
            drawMarker(camera, p, c, MARKER_CROSS, 10, 1);
            line(camera, Point(p.x - 8, p.y), Point(p.x + 8, p.y), c, 1);
            line(camera, Point(p.x, p.y - 8), Point(p.x, p.y + 8), c, 1);
            label(camera, "P" + to_string(i + 1) + " " + state, Point(max(0, p.x + 7), max(20, p.y + 15)), .45, c, 2);
        }

        // ------------------------------------------------------------ UI
        Mat ui(H, W, CV_8UC3, BG);
        fill(ui, Rect(0, 0, W, 76), Scalar(32, 36, 46));
        label(ui, "HAND-EYE  /  VISION-GUIDED STACKING", Point(20, 36), .83, WHITE, 2);

        label(ui, "ABB IRB120    |    OpenCV    |    8-bit serial", Point(22, 61), .43, MUTED);
        label(ui, calibration.lighting.text(), Point(570, 61), .38, GOLD);
        label(ui, "PLANNED / SENT ONLY - NO LIVE ABB FEEDBACK", Point(890, 43), .43, GOLD, 2);
        panel(ui, Rect(10, 86, 860, 648));
        Mat camFit;
        resize(camera, camFit, Size(CAMW, CAMH));
        camFit.copyTo(ui(a.imageRect));
        label(ui, "LIVE CAMERA   /   P1-P9", Point(25, 115), .5, WHITE, 2);
        panel(ui, Rect(878, 86, 392, 648));
        label(ui, "CYCLE CONTROL", Point(895, 116), .55, WHITE, 2);
        button(ui, Rect(1080, 92, 170, 34), a.setupOpen ? "CLOSE SETUP" : "SETUP", a.setupOpen);
        label(ui, "CHOOSE STACK PATTERN", Point(895, 147), .44, MUTED);
        for (int j = 0; j < 3; ++j)
            button(ui, Rect(887 + j * 124, 158, 118, 38), "S" + to_string(j + 1), a.selected == j + 1);
        int routine = a.selected;
        Plan p = a.planRefreshReady ? a.queued[routine - 1] : makePlan(a, routine);
        int active = -1;
        for (int j = 0; j < 3; ++j)
            if (a.sentPlans[j]) active = j;
        string phase = "READY TO START";
        if (active >= 0)
            phase = !a.returnSent[active] ? "PLAN SENT / WATCHING PICKUPS"
            : a.returnStarted[active] ? "RETURNING BLOCKS"
            : "RETURN PLAN SENT / WAITING";
        else if (!startBlock(a).empty())
            phase = "WAITING FOR SETUP / VISION";
        if (a.busy) phase = "SENDING PLAN";
        fill(ui, Rect(892, 210, 362, 55), Scalar(45, 66, 75));
        label(ui, phase, Point(901, 243), .46, GOLD, 2);
        label(ui, "TARGET S" + to_string(active >= 0 ? active + 1 : routine), { 895, 291 }, .38, WHITE, 2);
        label(ui, "LEVEL / PICKUP", { 1024, 291 }, .30, MUTED);
        label(ui, "OBSERVED", { 1184, 291 }, .28, MUTED);
        drawStack(ui, 895, 305, active >= 0 ? active + 1 : routine);
        for (int row = 0; row < 3; ++row) {
            int level = 2 - row;
            int src = active >= 0 ? a.sentSources[active][level] : p.source[level];
            label(ui,
                level == 2 ? "TOP L3"
                : level == 1 ? "MIDDLE L2"
                : "BOTTOM L1",
                { 1024, 318 + row * 34 }, .34, WHITE);
            label(ui, "PICK " + to_string(level + 1) + ": " + (src >= 0 ? "P" + to_string(src + 1) : "waiting"),
                { 1024, 332 + row * 34 }, .32, MUTED);
            int seen = active >= 0 ? a.stackVision.observedLayers[level] : -1;
            drawObservedColour(ui, { 1184, 305 + row * 34, 66, 31 }, seen);
        }
        label(ui, "Pickup order: L1 bottom -> L2 -> L3 top", { 895, 420 }, .35, MUTED);
        label(ui, "RANDOM RETURN: TOP FIRST", Point(895, 437), .45, WHITE, 2);
        for (int k = 0; k < 3; ++k) {
            int l = 2 - k;
            string target = active >= 0 && a.returnSent[active] ? "P" + to_string(a.returnDest[active][l] + 1)
                : "await empty slots";
            label(ui, "L" + to_string(l + 1) + "  >  " + target, Point(895, 463 + k * 24), .44, MUTED);
        }
        label(ui, a.serialConnected ? "COM: " + a.comName + " connected" : "COM: OFFLINE", Point(895, 553), .45,
            a.serialConnected ? ink(GREEN) : GOLD);
        button(ui, Rect(895, 574, 175, 30), "UNUSED ONLY", !a.allowRepeatPickups);
        button(ui, Rect(1078, 574, 175, 30), "ALLOW REPEATS", a.allowRepeatPickups);
        label(ui, "Camera-confirmed cycles: " + to_string(a.completedCycles) + (a.sentCount ? " / mode locked" : ""),
            Point(895, 625), .40, WHITE);
        string blocked = startBlock(a);
        button(ui, Rect(895, 640, 355, 58),
            blocked.empty() ? "START AUTOMATIC CYCLE"
            : a.sentCount ? "CYCLE RUNNING - PLEASE WAIT"
            : "START UNAVAILABLE",
            blocked.empty());
        if (!blocked.empty() && !a.sentCount) {
            std::istringstream words(blocked);
            string word, line;
            int y = 711;
            while (words >> word) {
                string next = line.empty() ? word : line + " " + word;
                if (getTextSize(next, FONT_HERSHEY_SIMPLEX, .34, 1, nullptr).width > 350 && !line.empty()) {
                    label(ui, line, { 895, y }, .34, GOLD);
                    y += 14;
                    line = word;
                    if (y > 725) break;
                }
                else
                    line = next;
            }
            if (y <= 725 && !line.empty()) label(ui, line, { 895, y }, .34, GOLD);
        }
        else
            label(ui, "S start    1 / 2 / 3 select    Q exit", Point(895, 723), .40, MUTED);
        panel(ui, Rect(10, 744, 1260, 124));
        if (a.setupOpen) {
            label(ui, "SETUP  /  changes allowed between cycles", Point(25, 774), .48, WHITE, 2);
            const array<string, 9> names = { {"M MAP", "V VERIFY", "R SAMPLE", "G SAMPLE", "B SAMPLE", "P REFRESH",
                                             "Z CALIBRATE", "X HISTORY", "K HSV RESET"} };
            for (int i = 0; i < 9; ++i) button(ui, Rect(25 + i * 138, 795, 129, 38), names[i]);
            button(ui, Rect(25, 833, 190, 28), "RESET STATIC MAP");
            button(ui, Rect(225, 833, 200, 28), "RESET STATIC SETUP");
            button(ui, Rect(435, 833, 220, 28), a.learnPick || a.learning ? "CANCEL LEARN" : "LEARN 40mm BLOCK");
            string measurement = a.learnedFace.width > 0
                ? to_string(cvRound(a.learnedFace.width)) + "x" +
                to_string(cvRound(a.learnedFace.height)) + " camera px (local reference)"
                : "No block size saved";
            if (a.learnedFace.width > 0 && a.learnedResolution != a.cameraSize)
                measurement = "Camera resolution changed - relearn size";
            if (measurement.size() > 48) measurement.resize(48);
            label(ui, measurement, Point(675, 853), .32, MUTED);
            button(ui, Rect(1060, 833, 190, 28), "NEW BATCH");
        }
        else {
            int empty = 0, unsure = 0;
            for (const auto& pos : a.positions) {
                unsure += pos.uncertain;
                empty += !pos.uncertain && pos.confirmed == UNKNOWN;
            }
            label(ui, "NEXT STEP", Point(25, 775), .47, GOLD, 2);
            string next =
                active < 0
                ? (blocked.empty() ? "Choose a pattern, then press START. The return runs automatically." : blocked)
                : !a.returnSent[active]
                ? "STACK MONITOR: " + a.stackVision.note
                : "Wait: the robot returns L3, L2, L1. Camera checks their colours at the assigned destinations.";
            label(ui, next, Point(25, 805), .43, WHITE);
            if (active >= 0 && !a.returnSent[active] &&
                chrono::duration<double>(chrono::steady_clock::now() - a.stackSentAt[active]).count() >= 30.0) {
                bool noneMoved = true;
                for (int src : a.sentSources[active])
                    if (a.positions[src].uncertain || a.positions[src].confirmed == UNKNOWN) noneMoved = false;
                if (noneMoved)
                    label(ui,
                        "NO PICKUP CHANGE: check RAPID running, RX BYTE log and dg input. No robot acknowledgement.",
                        Point(25, 834), .40, GOLD);
                else
                    label(ui, "Watching pickup changes; robot receipt is not confirmed by COM.", Point(25, 834), .42,
                        MUTED);
            }
            else
                label(ui,
                    "Empty slots: " + to_string(empty) + "    Uncertain: " + to_string(unsure) +
                    "    Camera updates every frame",
                    Point(25, 834), .45, MUTED);
            label(ui, "Use the physical robot STOP / emergency stop to stop motion.", Point(25, 856), .39, GOLD);
        }

        panel(ui, Rect(10, 876, 1260, 195));
        label(ui, "FIXED PICKUP TARGETS  /  P1-P9 (ABB TAUGHT POSITIONS)", Point(25, 899), .53, WHITE, 2);
        for (int i = 0; i < N; ++i) {
            int col = i % 9, x = 24 + col * 139, y = 916;
            Rect tile(x, y, 131, 131);
            const auto& st = a.positions[i];
            string state = a.mapped != N ? "UNMAPPED"
                : st.uncertain ? "UNCERTAIN"
                : st.confirmed == UNKNOWN ? "EMPTY"
                : cname(st.confirmed);
            Scalar color = state == "UNCERTAIN" ? GOLD : ink(st.confirmed);
            fill(ui, tile, Scalar(43, 49, 61));
            rectangle(ui, tile, a.mapped == N ? color : MUTED, 2);
            label(ui, "P" + to_string(i + 1), Point(x + 11, y + 25), .57, WHITE, 2);
            if (state == "RED" || state == "GREEN" || state == "BLUE")
                fill(ui, Rect(x + 41, y + 37, 49, 40), ink(st.confirmed));
            else
                rectangle(ui, Rect(x + 41, y + 37, 49, 40), MUTED, 1);
            if (a.processedBlocks[i])
                label(ui, a.allowRepeatPickups ? "USED - ALLOWED" : "USED - SKIP", Point(x + 8, y + 126), .32, GOLD, 1);
            label(ui, state, Point(x + 8, y + 107), state.size() > 7 ? .34 : .44, state == "EMPTY" ? MUTED : color, 1);
        }
        fill(ui, Rect(0, 1080, W, 40), Scalar(30, 34, 44));
        string msg = a.sampleMode ? string("CLICK ") + cname(a.sampleColour) + " BLOCK CENTRE TO CALIBRATE"
            : a.mapping ? "MAP: CLICK P" + to_string(a.mapped + 1)
            : a.message;
        if (!a.sentCount && !a.setupOpen && !a.mapping && !a.sampleMode && !blocked.empty())
            msg = "START BLOCKED: " + blocked;
        if (msg.size() > 110) msg.resize(110);
        label(ui, msg, Point(18, 1105), .48, GOLD);
        label(ui, BUILD_TAG, Point(W - 360, 1105), .4, MUTED);
        drawNavigation(ui, a);
        if (preview) {
            imwrite(a.setupOpen ? "operator_ui_setup_preview.png" : "operator_ui_preview.png", ui);
            break;
        }
        imshow("Hand Eye Robot Console", ui);

        int key = a.pendingKey >= 0 ? a.pendingKey : waitKey(25);
        a.pendingKey = -1;
        if (key < 0) continue;
        if (key == 'o' || key == 'O') {
            if (!a.sentCount) {
                a.setupOpen = !a.setupOpen;
                a.mapping = false;
                a.sampleMode = false;
                a.zoneSetup = 0;
                a.learnPick = false;
                a.learning = false;
            }
            continue;
        }
        if (a.sentCount && key != 'q' && key != 'Q' && key != 27) continue;
        if (!a.setupOpen && string("nNlLcC0dDtTzZxXmMvVrRgGbBkKpPuU").find(char(key)) != string::npos) continue;
        if (key == 'q' || key == 'Q' || key == 27) {
            if (requestAppExit(a)) break;
            continue;
        }
        if (key == 's' || key == 'S') {
            string reason = startBlock(a);
            if (reason.empty())
                operatorStart(a);
            else
                a.message = reason;
            continue;
        }
        if (key == 'n' || key == 'N') {
            newBatch(a);
            continue;
        }
        if (key == 'l' || key == 'L') {
            if (a.learnPick || a.learning) {
                a.learnPick = false;
                a.learning = false;
                a.sizeSamples.clear();
                a.message = "Measurement cancelled";
            }
            else {
                a.mapping = false;
                a.sampleMode = false;
                a.zoneSetup = 0;
                a.learnPick = true;
                a.message = "LEARN: place one 40x40mm face flat at pickup height, then click its outline";
            }
            continue;
        }
        if (key == 'c' || key == 'C' || key == '0') {
            operatorClear(a);
            continue;
        }
        if (key == 'd' || key == 'D') {
            operatorReturn(a);
            continue;
        }
        if (key == 't' || key == 'T') {
            operatorRelease(a);
            continue;
        }
        if (key == 'z' || key == 'Z') {
            switchPage(a, 1);
            continue;
        }
        if (a.zoneSetup == -1 && (key == '1' || key == '2' || key == '3')) {
            a.zoneSetup = key - '0';
            a.message = "Click S" + to_string(a.zoneSetup) + " stack zone centre";
            continue;
        }
        if (key == 'x' || key == 'X') {
            for (int j = 0; j < 3; ++j) {
                a.observedHistory[j].clear();
                a.stackTop[j] = 0;
                a.topCandidate[j] = 0;
                a.topStreak[j] = 0;
            }
            a.message = "Cleared stack observation history";
        }
        if (key == 'm' || key == 'M') {
            if (!a.planLocked) {
                a.learnPick = false;
                a.learning = false;
                a.processedBlocks.fill(false);
                a.autoProposal = false;
                a.mapping = true;
                a.mapped = 0;
                a.verifiedMap = false;
                a.autoCandidate = false;
                resetObservations(a);
                a.message = "Click ABB P1-P9 in order";
            }
        }
        if (key == 'v' || key == 'V') {
            if (a.mapped == N && !a.mapping) {
                a.verifiedMap = true;
                saveConfig(a);
                a.message = "Verified reference saved. Ensure labels match ABB taught P1-P9.";
            }
        }
        if (key == '1' || key == '2' || key == '3') {
            a.selected = key - '0';
            a.releaseArmed = false;
        }
        if (key == 'r' || key == 'R' || key == 'g' || key == 'G' || key == 'b' || key == 'B') {
            if (!a.planLocked) {
                a.sampleColour = (key == 'r' || key == 'R') ? RED : (key == 'g' || key == 'G') ? GREEN : BLUE;
                a.sampleMode = true;
            }
        }
        if (key == 'k' || key == 'K') {
            if (!a.planLocked) {
                a.hue = { {0, 0, 60, 110} };
                a.tolerance = { {0, 12, 15, 15} };
                a.satMin = 65;
                a.valMin = 60;
                resetObservations(a);
                saveConfig(a);
                a.message = "HSV defaults restored";
            }
        }
        if (key == 'p' || key == 'P') {
            if (a.mapped != N || a.mapping || !a.verifiedMap) {
                a.message = "Verify P1-P9 against ABB taught positions first (V or M)";
                continue;
            }
            refreshPlans(a);
            a.message = "VISION REFRESHED: choose any unused S, then send one stack";
        }
        if (key == 'u' || key == 'U') {
            a.planLocked = false;
            a.planRefreshReady = false;
            resetObservations(a);
            a.message = "UI refreshed; reservations retained (use T after robot returns blocks).";
        }
    }
    if (a.port) operatorClear(a); // final DIO clear; cannot stop a running robot
    serialVisionPump = {};
    disconnect(a);
    cap.release();
    destroyAllWindows();
    return 0;
}
