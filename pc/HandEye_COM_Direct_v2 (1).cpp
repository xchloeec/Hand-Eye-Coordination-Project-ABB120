// Hand-Eye HD dashboard: stack blocks, then drop them at RANDOM free positions. Repeatable.
// C++14 or newer, OpenCV 4, libserialport. Windows: link opencv_world4xx.lib and serialport.lib.
// Run: HandEye_COM_Direct.exe COM3
//
// CYCLE (repeat as often as needed):
//   1/2/3 select stack -> S send stack plan -> robot stacks without pendant input
//   -> robot moves home -> camera finds empty slots -> automatic random return
//   -> camera confirms returned colours -> automatic release -> ready for next stack.
// Other keys: M map | V verify | P refresh plans | R/G/B sample colour | K reset HSV
//             Z then 1/2/3 then click = stack zone | X clear history | Q quit
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
using namespace cv;
using namespace std;

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
        bool exists(const string& path) { ifstream file(path, ios::binary); return file.good(); }
    }


    using cv::Point2d;
    using H = cv::Matx33d;
    constexpr int UIW = 1280, UIH = 900, STABLE_FRAMES = 20;
    constexpr double RESET_JUMP_PX = 12.0, DRIFT_LIMIT_PX = 3.0; // relaxed: fit gate (2 mm) is still enforced
    const cv::Rect VIDEO(15, 100, 820, 615);
    const cv::Scalar BG(30, 26, 23), PANEL(48, 43, 38), FG(240, 240, 235), MUTED(185, 180, 170), GOLD(85, 205, 245);
    const char* TITLE = "Block Calibration | Preview Only | NO ROBOT CONNECTION";
    std::string fmt(double v, int n = 2) { std::ostringstream s;s << std::fixed << std::setprecision(n) << v;return s.str(); }
    std::string stamp() {
        auto now = std::chrono::system_clock::now();std::time_t t = std::chrono::system_clock::to_time_t(now);std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char b[40];std::strftime(b, sizeof(b), "%Y%m%d_%H%M%S", &tm);return b;
    }
    Point2d project(const H& h, Point2d p) { cv::Vec3d v = h * cv::Vec3d(p.x, p.y, 1);if (std::abs(v[2]) < 1e-10)throw std::runtime_error("Mapping denominator is zero");return { v[0] / v[2],v[1] / v[2] }; }
    std::vector<Point2d> localGrid() { std::vector<Point2d> p;for (int r = 0;r < 3;++r)for (int c = 0;c < 3;++c)p.emplace_back(40 + 75 * c, 180 - 70 * r);return p; }
    std::vector<Point2d> nominalRobot() { auto p = localGrid();for (auto& a : p)a = { 552.65 - a.y,a.x - 122.80 };return p; }
    double polygonArea(const std::vector<Point2d>& p) { std::vector<cv::Point2f> q;for (auto a : p)q.emplace_back(float(a.x), float(a.y));std::vector<cv::Point2f> hull;cv::convexHull(q, hull);return std::abs(cv::contourArea(hull)); }
    bool fitH(const std::vector<Point2d>& src, const std::vector<Point2d>& dst, H& result) {
        if (src.size() < 4 || src.size() != dst.size() || polygonArea(src) < 10 || polygonArea(dst) < 10)return false;
        auto norm = [](const std::vector<Point2d>& p) {Point2d mean{};for (auto v : p)mean += v;mean *= 1. / p.size();double d = 0;for (auto v : p)d += cv::norm(v - mean);d /= p.size();double s = std::sqrt(2.) / std::max(d, 1e-12);return H(s, 0, -s * mean.x, 0, s, -s * mean.y, 0, 0, 1);};
        H a = norm(src), b = norm(dst);cv::Mat m(int(src.size() * 2), 9, CV_64F, cv::Scalar(0));
        for (size_t i = 0;i < src.size();++i) { auto u = project(a, src[i]), x = project(b, dst[i]);double r0[] = { -u.x,-u.y,-1,0,0,0,x.x * u.x,x.x * u.y,x.x };double r1[] = { 0,0,0,-u.x,-u.y,-1,x.y * u.x,x.y * u.y,x.y };for (int j = 0;j < 9;++j) { m.at<double>(int(i * 2), j) = r0[j];m.at<double>(int(i * 2 + 1), j) = r1[j]; } }
        cv::SVD svd(m, cv::SVD::FULL_UV);if (svd.w.rows < 8 || svd.w.at<double>(7) < svd.w.at<double>(0) * 1e-7)return false;
        H h;for (int i = 0;i < 9;++i)h.val[i] = svd.vt.at<double>(8, i);h = b.inv() * h * a;
        if (std::abs(h(2, 2)) < 1e-10)return false;h *= 1. / h(2, 2);if (!cv::checkRange(cv::Mat(h)))return false;result = h;return true;
    }
    struct ColourModel { int sat = 55, val = 35, minArea = 200;std::array<int, 3> hue{ 0,60,110 }, tol{ 13,25,20 }; };
    struct Detection { Point2d centre;cv::Rect box;std::vector<cv::Point> quad;int colour = 0;double area = 0; };
    Point2d crossCentre(const std::vector<cv::Point>& q) { Point2d a = q[0], b = q[2], c = q[1], d = q[3], r = b - a, s = d - c;double den = r.x * s.y - r.y * s.x;if (std::abs(den) < 1e-9)return (a + b + c + d) * .25;double t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / den;return a + t * r; }
    std::vector<Detection> detect(const cv::Mat& image, const ColourModel& cfg) {
        cv::Mat hsv;cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);std::vector<Detection> out;
        for (int k = 0;k < 3;++k) {
            cv::Mat mask(hsv.size(), CV_8U, cv::Scalar(0)), part;
            for (int shift = -180;shift <= 180;shift += 180) { int lo = std::max(0, cfg.hue[k] - cfg.tol[k] + shift), hi = std::min(179, cfg.hue[k] + cfg.tol[k] + shift);if (lo <= hi) { cv::inRange(hsv, cv::Scalar(lo, cfg.sat, cfg.val), cv::Scalar(hi, 255, 255), part);mask |= part; } }
            cv::medianBlur(mask, mask, 3);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, { 3,3 }));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, { 3,3 }));
            std::vector<std::vector<cv::Point>> contours;cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (auto& contour : contours) {
                double area = cv::contourArea(contour);if (area < cfg.minArea)continue;
                std::vector<cv::Point> hull, q;cv::convexHull(contour, hull);
                double hullArea = cv::contourArea(hull);
                // Fill small segmentation notches, but reject concave/merged shapes.
                if (hullArea <= 0 || area / hullArea < .85)continue;
                // Rounded/blurred square corners reduce quad coverage to about 0.78-0.85.
                // Require convexity/solidity and edge aspect too; do not force rectangles onto arbitrary contours.
                bool quad = false;
                for (double epsilon : {.018, .025, .035, .045, .055, .065, .075}) {
                    cv::approxPolyDP(hull, q, cv::arcLength(hull, true) * epsilon, true);
                    if (q.size() == 4 && cv::isContourConvex(q)) {
                        double qa = std::abs(cv::contourArea(q));
                        if (qa / hullArea >= .75 && qa / hullArea <= 1.05) { quad = true;break; }
                    }
                }
                if (!quad)continue;
                double minEdge = 1e9, maxEdge = 0;
                for (int j = 0;j < 4;++j) { double e = cv::norm(q[j] - q[(j + 1) % 4]);minEdge = std::min(minEdge, e);maxEdge = std::max(maxEdge, e); }
                if (minEdge < 8 || maxEdge / minEdge>2.0)continue;
                cv::Rect box = cv::boundingRect(q);if (box.x <= 1 || box.y <= 1 || box.br().x >= image.cols - 1 || box.br().y >= image.rows - 1)continue;
                if (area < box.area() * .45)continue;
                out.push_back({ crossCentre(q),box,q,k,area });
            }
        }
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) {if (a.centre.y != b.centre.y)return a.centre.y < b.centre.y;return a.centre.x < b.centre.x;});return out;
    }
    bool gridOrder(std::vector<Detection>& d, bool flipRows, bool flipCols) {
        if (d.size() != 9)return false;
        std::sort(d.begin(), d.end(), [](auto& a, auto& b) {return a.centre.y < b.centre.y;});
        double meanSize = 0;for (auto& a : d)meanSize += std::sqrt(a.area);meanSize /= 9;
        for (int r = 0;r < 3;++r) {
            std::sort(d.begin() + r * 3, d.begin() + r * 3 + 3, [](auto& a, auto& b) {return a.centre.x < b.centre.x;});if (d[r * 3 + 2].centre.x - d[r * 3].centre.x < meanSize)return false;
            if (r && d[r * 3].centre.y - d[(r - 1) * 3].centre.y < meanSize * .5)return false;
        }
        for (int i = 0;i < 9;++i)for (int j = i + 1;j < 9;++j)if (cv::norm(d[i].centre - d[j].centre) < meanSize * .7)return false;
        auto ordered = d;for (int r = 0;r < 3;++r)for (int c = 0;c < 3;++c)d[r * 3 + c] = ordered[(flipRows ? 2 - r : r) * 3 + (flipCols ? 2 - c : c)];return true;
    }
    struct Mapping { H h = H::eye();std::vector<Point2d> pixels, robot;cv::Size resolution;double rms = 0, max = 0; };
    bool makeMapping(const std::vector<Detection>& d, const std::vector<Point2d>& robot, cv::Size resolution, Mapping& m) { if (d.size() != 9)return false;std::vector<Point2d> p;for (auto& a : d)p.push_back(a.centre);if (!fitH(p, robot, m.h))return false;m.pixels = p;m.robot = robot;m.resolution = resolution;m.rms = 0;m.max = 0;for (int i = 0;i < 9;++i) { double e = cv::norm(project(m.h, p[i]) - robot[i]);m.rms += e * e;m.max = std::max(m.max, e); }m.rms = std::sqrt(m.rms / 9);return true; }
    double regionDistance(const Mapping& m, Point2d robot) { std::vector<cv::Point2f> hull;for (int i : {0, 2, 8, 6})hull.emplace_back(float(m.robot[i].x), float(m.robot[i].y));return cv::pointPolygonTest(hull, { float(robot.x),float(robot.y) }, true); }
    bool inRegion(const Mapping& m, Point2d robot) { return regionDistance(m, robot) >= 0; }
    bool nearBoundary(const Mapping& m, Point2d robot) { return std::abs(regionDistance(m, robot)) <= 2.0; } // display only; does not expand planning region
    std::string regionLabel(const Mapping& m, Point2d p) { return nearBoundary(m, p) ? "EDGE (within 2mm)" : inRegion(m, p) ? "INSIDE" : "OUTSIDE"; }
    struct Options { std::string source = "camera", image, reference = "reference_points.csv";int camera = 0;double fitLimit = 2;bool explicitReference = false, selftest = false, snapshot = false; };
    struct Button { cv::Rect r;std::string label;char action; };
    struct Plan { bool valid = false;std::array<int, 3> source{ -1,-1,-1 };std::array<Point2d, 3> xy{};std::array<int, 3> colours{};std::string reason; };
    // Same top-to-bottom patterns as the original planning main.cpp; detected colours use 0-based IDs here.
    const std::array<std::array<int, 3>, 3> PATTERNS{ {{{1,0,2}},{{0,1,2}},{{2,1,0}}} };
    const char* COLOUR_NAMES[] = { "RED","GREEN","BLUE" };
    Plan makePlan(const std::vector<Detection>& detections, const Mapping& mapping, int routine) {
        Plan p;if (routine < 1 || routine>3) { p.reason = "Invalid pattern";return p; }
        for (int l = 0;l < 3;++l) {
            int wanted = PATTERNS[routine - 1][2 - l];p.colours[l] = wanted;
            for (int j = 0;j<int(detections.size());++j) {
                if (detections[j].colour != wanted)continue;bool used = false;for (int k = 0;k < l;++k)if (p.source[k] == j)used = true;if (used)continue;
                Point2d xy = project(mapping.h, detections[j].centre);if (!inRegion(mapping, xy))continue;
                p.source[l] = j;p.xy[l] = xy;break;
            }
            if (p.source[l] < 0) { p.reason = std::string("Need in-region ") + COLOUR_NAMES[wanted];return p; }
        }
        p.valid = true;p.reason = "Preview ready; not a motion approval";return p;
    }
    struct App {
        Options opt;ColourModel cfg;cv::Mat frame, ui;std::vector<Detection> detections, grid, previous;std::vector<Point2d> refs = nominalRobot();Mapping mapping;
        bool test = false, valid = false, labelsOK = false, refsOK = false, planeOK = false, flipRows = false, flipCols = false, freeze = false, missing = false, noise = false, exit = false, selectStable = false;
        struct CompareRow { Point2d reference, predicted;double error; };
        std::vector<CompareRow> comparison;bool comparisonView = false;
        std::string comparisonPath;
        bool rawView = false;
        double rawPointStep = 0, rawPointDrift = 0;
        std::deque<std::vector<Point2d>> pointHistory;
        std::vector<Point2d> lastRawCentres;
        double stableSpread = 0, windowDrift = 0, noiseLimitPx = 4.0;
        bool gridQualityReady = false;
        std::string stabilityReason = "Need nine fresh detections";
        static double median(std::vector<double> v) { std::sort(v.begin(), v.end());size_t n = v.size();return (v[(n - 1) / 2] + v[n / 2]) * .5; }
        std::vector<Point2d> averagedCentres()const {
            if (pointHistory.empty())return {};
            std::vector<Point2d> out(9);
            for (int i = 0;i < 9;++i) { std::vector<double>x, y;for (const auto& f : pointHistory) { x.push_back(f[i].x);y.push_back(f[i].y); }out[i] = { median(x),median(y) }; }
            return out;
        }
        void updateGridStability() {
            gridQualityReady = false;rawPointStep = 0;rawPointDrift = 0;stableSpread = 0;windowDrift = 0;
            if (!gridOrder(grid, flipRows, flipCols)) {
                stable = 0;previous.clear();grid.clear();pointHistory.clear();lastRawCentres.clear();
                stabilityReason = "Need nine fresh separate tops";return;
            }
            bool same = previous.size() == 9;std::vector<Point2d> raw;
            for (int i = 0;i < 9;++i) {
                raw.push_back(grid[i].centre);
                if (lastRawCentres.size() == 9)rawPointStep = std::max(rawPointStep, cv::norm(raw.back() - lastRawCentres[i]));
                if (previous.size() == 9) { double drift = cv::norm(previous[i].centre - raw.back());rawPointDrift = std::max(rawPointDrift, drift);if (drift > RESET_JUMP_PX || previous[i].colour != grid[i].colour)same = false; }
            }
            if (!same) { previous = grid;pointHistory.clear(); }
            pointHistory.push_back(raw);if (pointHistory.size() > STABLE_FRAMES)pointHistory.pop_front();lastRawCentres = raw;
            stable = int(pointHistory.size());stabilityReason = "Collecting fresh 20-frame window";
            if (pointHistory.size() < STABLE_FRAMES)return;
            auto centre = averagedCentres();bool acceptable = true;
            for (int i = 0;i < 9;++i) {
                std::vector<double>dist, firstX, firstY, lastX, lastY;
                for (size_t j = 0;j < pointHistory.size();++j) {
                    auto p = pointHistory[j][i];dist.push_back(cv::norm(p - centre[i]));
                    if (j < 5) { firstX.push_back(p.x);firstY.push_back(p.y); }if (j >= 15) { lastX.push_back(p.x);lastY.push_back(p.y); }
                }
                std::sort(dist.begin(), dist.end());double spread = dist[17]; // at least 18/20 samples within 2.5px
                auto mean = [](const std::vector<double>& v) {return std::accumulate(v.begin(), v.end(), 0.0) / v.size();};
                double drift = cv::norm(Point2d(mean(firstX), mean(firstY)) - Point2d(mean(lastX), mean(lastY)));
                stableSpread = std::max(stableSpread, spread);windowDrift = std::max(windowDrift, drift);
                if (spread > noiseLimitPx || drift > DRIFT_LIMIT_PX)acceptable = false;
            }
            gridQualityReady = acceptable;
            if (!acceptable) { stabilityReason = windowDrift > DRIFT_LIMIT_PX ? "Window drifting - keep blocks/camera still" : "Point noise too high - sample colours/check light"; }
            else stabilityReason = "20-frame window stable (median centres)";
        }
        int stable = 0, selected = 0, pose = 0, trial = 0, sample = -1, snapshotNo = 0, testStable = 0;Point2d testPrevious{}, block{ 105,125 };
        bool entering = false, planning = false, planLocked = false;int pattern = 1, planStable = 0;Plan lockedPlan;std::vector<Detection> planAnchor;
        std::string input;Point2d measuredPixel{}, measuredPrediction{};cv::Mat validationFrame;
        std::string session;std::string message = "Place nine cubes. Confirm P1-P9 direction and measured robot reference coordinates.";
        std::vector<Button> buttons = { {{15,50,165,34},"1 Calibration",'1'},{{190,50,180,34},"2 Coordinate test",'2'},{{380,50,160,34},"3 Planning",'3'},{{550,50,130,34},"C Fit + save",'c'},{{690,50,130,34},"I Invalidate",'i'},{{830,50,130,34},"S Snapshot",'s'},{{970,50,170,34},"E Actual XY",'e'},{{1150,50,110,34},"Q Quit",'q'} };
        std::string sourceName()const { return opt.source == "camera" ? "LIVE CAMERA " + std::to_string(opt.camera) : opt.source == "demo" ? "SYNTHETIC DEMO" : "IMAGE FILE"; }
        void invalidate(const std::string& why) { valid = false;labelsOK = refsOK = planeOK = false;stable = 0;comparisonView = false;gridQualityReady = false;previous.clear();pointHistory.clear();lastRawCentres.clear();entering = false;planLocked = false;planStable = 0;planAnchor.clear();message = why; }
        std::string sessionDir() { if (session.empty()) { auto id = std::chrono::system_clock::now().time_since_epoch().count();fileutil::createDirectory("sessions");session = "sessions/" + (stamp() + "_" + opt.source + "_" + std::to_string(id));fileutil::createDirectory(session); }return session; }
        void update() {
            if (valid && frame.size() != mapping.resolution)invalidate("Resolution changed - recalibrate.");
            detections = detect(frame, cfg);grid = detections;
            bool unchanged = planAnchor.size() == detections.size() && !detections.empty();
            if (unchanged)for (size_t i = 0;i < detections.size();++i)if (planAnchor[i].colour != detections[i].colour || cv::norm(planAnchor[i].centre - detections[i].centre) > 2.5)unchanged = false;
            if (unchanged)planStable = std::min(1000, planStable + 1);else { planStable = 1;planAnchor = detections;if (planLocked) { planLocked = false;message = "Scene changed - preview unlocked. Wait for stable detections."; } }
            updateGridStability();
            if (selected >= int(detections.size()))selected = 0;
            if (!detections.empty()) { auto p = detections[selected].centre;if (cv::norm(p - testPrevious) < 2.5)testStable++;else testStable = 0;testPrevious = p;selectStable = testStable >= 10; }
            else { testStable = 0;selectStable = false; }
        }
        Plan currentPlan()const { if (!valid) { Plan p;p.reason = "Calibrate first";return p; }if (planStable < STABLE_FRAMES) { Plan p;p.reason = "Wait for stable block positions";return p; }return planLocked ? lockedPlan : makePlan(detections, mapping, pattern); }
        void savePlan() {
            Plan p = currentPlan();if (!planning || !p.valid) { message = "Open Planning and wait for a valid stable plan.";return; }auto dir = sessionDir();std::string name = "planning_preview_" + std::to_string(++snapshotNo) + ".csv";std::ofstream f(dir + "/" + name);if (!f)throw std::runtime_error("Cannot save planning preview");
            f << "source,stack,level_bottom_first,colour,detection_id,pick_robot_x_mm,pick_robot_y_mm,status\n" << std::setprecision(12);
            for (int l = 0;l < 3;++l)f << opt.source << ",S" << pattern << ',' << l + 1 << ',' << COLOUR_NAMES[p.colours[l]] << ",B" << p.source[l] + 1 << ',' << p.xy[l].x << ',' << p.xy[l].y << ",PREVIEW_ONLY_NO_MOTION\n";
            message = "Saved " + name + ". No commands sent; fixed S" + std::to_string(pattern) + " remains on the robot side.";
        }
        void calibration() {
            if (test) { message = "Switch to Calibration first.";return; }
            if (grid.size() != 9) { message = "BLOCKED: detected " + std::to_string(detections.size()) + " tops; need nine separate tops";return; }
            if (stable < STABLE_FRAMES) { message = "COLLECTING: " + std::to_string(stable) + "/20 fresh frames";return; }
            if (!gridQualityReady) {
                message = windowDrift > DRIFT_LIMIT_PX ? "BLOCKED: window drift " + fmt(windowDrift, 1) + "px exceeds " + fmt(DRIFT_LIMIT_PX, 1) + "px; keep camera/blocks still" :
                    "BLOCKED: noise spread " + fmt(stableSpread, 1) + "px exceeds limit " + fmt(noiseLimitPx, 1) + "px; adjust Noise +/- or improve detection";return;
            }
            if (!labelsOK || !refsOK || !planeOK) { message = "BLOCKED: confirm label direction, robot references, and same cube-top plane (V / W / H).";return; }
            auto fitPoints = grid;auto mean = averagedCentres();
            if (pointHistory.size() < STABLE_FRAMES || mean.size() != 9) { message = "Need 20 fresh stable frames before fitting";return; }
            for (int i = 0;i < 9;++i)fitPoints[i].centre = mean[i];
            Mapping candidate;
            if (!makeMapping(fitPoints, refs, frame.size(), candidate)) { valid = false;message = "REJECTED: mapping could not be calculated; check point ordering and reference geometry";return; }
            if (candidate.max > opt.fitLimit) {
                valid = false;int worst = 0;double worstError = -1;
                for (int i = 0;i < 9;++i) { double e = cv::norm(project(candidate.h, fitPoints[i].centre) - refs[i]);if (e > worstError) { worstError = e;worst = i; } }
                message = "REJECTED: max " + fmt(candidate.max, 2) + "mm at P" + std::to_string(worst + 1) + ", RMS " + fmt(candidate.rms, 2) + "mm (limit " + fmt(opt.fitLimit, 2) + "). Check labels/actual robot XY.";
                try {
                    auto dir = sessionDir();std::string tag = "rejected_fit_" + std::to_string(++snapshotNo);
                    std::ofstream csv(dir + "/" + tag + ".csv");if (!csv)throw std::runtime_error("cannot write rejection CSV");
                    csv << "label,pixel_u,pixel_v,reference_x_mm,reference_y_mm,fitted_x_mm,fitted_y_mm,residual_mm\n";
                    for (int i = 0;i < 9;++i) { auto predicted = project(candidate.h, fitPoints[i].centre);csv << "P" << i + 1 << ',' << fitPoints[i].centre.x << ',' << fitPoints[i].centre.y << ',' << refs[i].x << ',' << refs[i].y << ',' << predicted.x << ',' << predicted.y << ',' << cv::norm(predicted - refs[i]) << '\n'; }
                    csv.flush();if (!csv)throw std::runtime_error("rejection CSV write failed");
                    if (!cv::imwrite(dir + "/" + tag + "_raw.png", frame))throw std::runtime_error("cannot write rejection image");
                    std::cout << message << " Evidence: " << dir << "/" << tag << ".csv" << std::endl;
                }
                catch (const std::exception& e) { message += " (evidence save failed)";std::cerr << e.what() << std::endl; }
                return;
            }
            mapping = candidate;valid = true;session.clear();
            try { saveCalibration();savePersistentCalibration();message = "Saved. Loads automatically next start (calibration_saved.yml) - no recalibration needed. Fit is NOT physical accuracy; test NEW positions."; }
            catch (const std::exception& e) { valid = false;message = std::string("SAVE FAILED: ") + e.what(); }
        }
        void saveCalibration() {
            auto dir = sessionDir();cv::FileStorage f((dir + "/" + "calibration.yml"), cv::FileStorage::WRITE);if (!f.isOpened())throw std::runtime_error("Cannot write calibration.yml");
            f << "schema" << 1 << "source" << sourceName() << "simulation" << int(opt.source == "demo") << "physically_validated" << 0 << "lens_undistorted" << 0 << "time_local" << stamp() << "width" << frame.cols << "height" << frame.rows;
            f << "H_pixel_to_robot_xy" << cv::Mat(mapping.h) << "fit_rms_mm" << mapping.rms << "fit_max_mm" << mapping.max << "fit_gate_mm" << opt.fitLimit;
            f << "operator_confirmed_labels" << int(labelsOK) << "operator_confirmed_robot_references" << int(refsOK) << "operator_confirmed_same_top_plane" << int(planeOK) << "flip_rows" << int(flipRows) << "flip_columns" << int(flipCols);
            f << "hsv_min_saturation" << cfg.sat << "hsv_min_value" << cfg.val << "min_contour_area_px" << cfg.minArea;
            f << "hue_centres" << "[";for (auto h : cfg.hue)f << h;f << "]" << "hue_tolerances" << "[";for (auto t : cfg.tol)f << t;f << "]";
            f << "averaged_frames" << int(pointHistory.size()) << "raw_jump_reset_px" << RESET_JUMP_PX << "window_90pct_spread_limit_px" << noiseLimitPx << "window_drift_limit_px" << DRIFT_LIMIT_PX << "centre_estimator" << "coordinate_median";
            f << "cube_height_nominal_mm" << 36.0 << "points" << "[";for (int i = 0;i < 9;++i)f << "{" << "label" << "P" + std::to_string(i + 1) << "pixel_u" << mapping.pixels[i].x << "pixel_v" << mapping.pixels[i].y << "robot_x_mm" << refs[i].x << "robot_y_mm" << refs[i].y << "}";f << "]";f.release();
            std::ofstream csv(dir + "/" + "calibration_points.csv");if (!csv)throw std::runtime_error("Cannot write points CSV");csv << "label,pixel_u,pixel_v,reference_robot_x_mm,reference_robot_y_mm,predicted_x_mm,predicted_y_mm,fit_error_mm\n" << std::setprecision(12);
            for (int i = 0;i < 9;++i) { auto p = project(mapping.h, mapping.pixels[i]);csv << "P" << i + 1 << ',' << mapping.pixels[i].x << ',' << mapping.pixels[i].y << ',' << refs[i].x << ',' << refs[i].y << ',' << p.x << ',' << p.y << ',' << cv::norm(p - refs[i]) << '\n'; }
            cv::imwrite((dir + "/" + "calibration_raw.png"), frame);draw();cv::imwrite((dir + "/" + "calibration_ui.png"), ui);
        }
        void savePersistentCalibration() {
            cv::FileStorage f("calibration_saved.yml", cv::FileStorage::WRITE);if (!f.isOpened())throw std::runtime_error("Cannot write calibration_saved.yml");
            f << "schema" << 2 << "width" << mapping.resolution.width << "height" << mapping.resolution.height << "H" << cv::Mat(mapping.h) << "rms" << mapping.rms << "max" << mapping.max;
            f << "flip_rows" << int(flipRows) << "flip_columns" << int(flipCols) << "sat" << cfg.sat << "val" << cfg.val << "min_area" << cfg.minArea;
            f << "hue" << "[";for (auto h : cfg.hue)f << h;f << "]" << "tol" << "[";for (auto t : cfg.tol)f << t;f << "]";
            f << "pixels" << "[";for (auto& q : mapping.pixels)f << q.x << q.y;f << "]" << "robot" << "[";for (auto& q : mapping.robot)f << q.x << q.y;f << "]";
            f.release();
        }
        // Reload the last accepted fit so the operator does not recalibrate on every start.
        bool loadPersistentCalibration() {
            if (!fileutil::exists("calibration_saved.yml"))return false;
            try {
                cv::FileStorage f("calibration_saved.yml", cv::FileStorage::READ);if (!f.isOpened())return false;
                int w = 0, h = 0;cv::Mat hm;std::vector<double> px, rb;std::vector<int> hue, tol;Mapping m;int sat = 0, val = 0, minArea = 0, fr = 0, fc = 0;
                f["width"] >> w;f["height"] >> h;f["H"] >> hm;f["pixels"] >> px;f["robot"] >> rb;f["hue"] >> hue;f["tol"] >> tol;f["rms"] >> m.rms;f["max"] >> m.max;
                f["sat"] >> sat;f["val"] >> val;f["min_area"] >> minArea;f["flip_rows"] >> fr;f["flip_columns"] >> fc;
                if (w <= 0 || h <= 0 || hm.rows != 3 || hm.cols != 3 || px.size() != 18 || rb.size() != 18 || hue.size() != 3 || tol.size() != 3)return false;
                hm.convertTo(hm, CV_64F);
                for (int r = 0;r < 3;++r)for (int c = 0;c < 3;++c) { m.h(r, c) = hm.at<double>(r, c);if (!std::isfinite(m.h(r, c)))return false; }
                for (int i = 0;i < 9;++i) { m.pixels.emplace_back(px[2 * i], px[2 * i + 1]);m.robot.emplace_back(rb[2 * i], rb[2 * i + 1]); }
                m.resolution = cv::Size(w, h);
                for (int i = 0;i < 9;++i)if (cv::norm(project(m.h, m.pixels[i]) - m.robot[i]) > 5.0)return false;
                mapping = m;refs = m.robot;flipRows = fr != 0;flipCols = fc != 0;cfg.sat = sat;cfg.val = val;cfg.minArea = minArea;
                for (int i = 0;i < 3;++i) { cfg.hue[i] = hue[i];cfg.tol[i] = tol[i]; }
                valid = true;labelsOK = refsOK = planeOK = true;
                message = "Loaded saved calibration (" + std::to_string(w) + "x" + std::to_string(h) + ", fit max " + fmt(m.max, 2) + " mm). No recalibration needed unless the camera moved.";
                return true;
            }
            catch (...) { return false; }
        }
        void snapshot() { auto dir = sessionDir();std::string tag = "capture_" + std::to_string(++snapshotNo);if (!cv::imwrite((dir + "/" + (tag + "_raw.png")), frame) || !cv::imwrite((dir + "/" + (tag + "_ui.png")), ui))throw std::runtime_error("Snapshot failed");message = "Saved raw + annotated snapshot in " + dir; }
        void compareNine() {
            if (!valid || !test) { message = "Save calibration, open Coordinate Test, then use 9-point comparison";return; }
            auto ordered = detections;
            if (!gridOrder(ordered, flipRows, flipCols)) { message = "9-point check requires the original nine-block arrangement; restore all nine blocks to P1-P9";return; }
            comparison.clear();
            for (int i = 0;i < 9;++i) { auto predicted = project(mapping.h, ordered[i].centre);comparison.push_back({ refs[i],predicted,cv::norm(predicted - refs[i]) }); }
            try {
                auto dir = sessionDir();std::string tag = "nine_point_comparison_" + std::to_string(++snapshotNo);comparisonPath = dir + "/" + tag + ".csv";
                std::ofstream csv(comparisonPath);if (!csv)throw std::runtime_error("Cannot write comparison CSV");
                csv << "label,code_x_mm,code_y_mm,camera_x_mm,camera_y_mm,delta_x_mm,delta_y_mm,xy_difference_mm,pixel_u,pixel_v\n" << std::setprecision(12);
                for (int i = 0;i < 9;++i) {
                    auto r = comparison[i];auto delta = r.predicted - r.reference;
                    csv << "P" << i + 1 << ',' << r.reference.x << ',' << r.reference.y << ',' << r.predicted.x << ',' << r.predicted.y << ',' << delta.x << ',' << delta.y << ',' << r.error << ',' << ordered[i].centre.x << ',' << ordered[i].centre.y << '\n';
                }
                csv.flush();if (!csv)throw std::runtime_error("Comparison write failed");
                if (!cv::imwrite(dir + "/" + tag + "_raw.png", frame))throw std::runtime_error("Comparison snapshot failed");
                comparisonView = true;freeze = true;
                message = "9-point comparison saved/frozen. X returns live. These are calibration-reference differences, not independent accuracy.";
            }
            catch (const std::exception& e) { message = e.what(); }
        }
        void beginMeasurement() { if (!test || !valid || detections.empty() || !selectStable) { message = "Need a valid mapping and a stationary selected block on Coordinate Test.";return; }auto p = detections[selected].centre;auto pred = project(mapping.h, p);if (!inRegion(mapping, pred)) { message = "Selected centre is outside calibrated region.";return; }measuredPixel = p;measuredPrediction = pred;validationFrame = frame.clone();input.clear();entering = true;freeze = true;message = "Enter independently measured ABB X,Y in mm; Enter saves, Esc cancels."; }
        void finishMeasurement() {
            std::istringstream s(input);double x, y;char comma;std::string extra;if (!(s >> x >> comma >> y) || comma != ',' || (s >> extra) || !std::isfinite(x) || !std::isfinite(y) || std::abs(x) > 10000 || std::abs(y) > 10000) { message = "Invalid value. Example: 420.50,-12.25 . Esc cancels.";return; }
            Point2d actual{ x,y };double error = cv::norm(actual - measuredPrediction);auto dir = sessionDir();auto path = dir + "/" + "independent_checks.csv";bool exists = fileutil::exists(path);std::ofstream f(path, std::ios::app);if (!f)throw std::runtime_error("Cannot write validation CSV");
            if (!exists)f << "time_local,source,pixel_u,pixel_v,predicted_x_mm,predicted_y_mm,measured_x_mm,measured_y_mm,error_mm,evidence_image\n";
            std::string filename = "check_" + std::to_string(++snapshotNo) + "_raw.png";cv::imwrite((dir + "/" + filename), validationFrame);
            f << std::setprecision(12) << stamp() << ',' << opt.source << ',' << measuredPixel.x << ',' << measuredPixel.y << ',' << measuredPrediction.x << ',' << measuredPrediction.y << ',' << x << ',' << y << ',' << error << ',' << filename << '\n';
            entering = false;freeze = false;message = "Independent check saved: error " + fmt(error) + " mm. This does not automatically approve robot motion.";
        }
        void sampleAt(cv::Point p) {
            if (sample < 0 || frame.empty())return;cv::Mat hsv;cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);cv::Rect patch(p.x - 5, p.y - 5, 11, 11);patch &= cv::Rect(0, 0, frame.cols, frame.rows);double x = 0, y = 0;int count = 0;for (int r = patch.y;r < patch.br().y;++r)for (int c = patch.x;c < patch.br().x;++c) { auto a = hsv.at<cv::Vec3b>(r, c);if (a[1] > 50 && a[2] > 40) { double angle = a[0] * CV_PI / 90.;x += std::cos(angle);y += std::sin(angle);++count; } }
            if (count < 10) { message = "Sample failed: click a clear coloured top face.";return; }double angle = std::atan2(y, x);if (angle < 0)angle += 2 * CV_PI;cfg.hue[sample] = int(std::round(angle * 90 / CV_PI)) % 180;cfg.tol[sample] = 13;sample = -1;invalidate("Colour sampled. Check detections and reconfirm calibration.");
        }
        void action(char k) {
            switch (k) {
            case '1':test = false;planning = false;planLocked = false;entering = false;freeze = false;message = "Calibration page. Arrange nine cubes and verify labels.";break;
            case '2':if (!valid) { message = "Fit and save calibration before Coordinate Test.";break; }test = true;planning = false;planLocked = false;freeze = false;testStable = 0;message = "Remove grid without moving camera/support plane. Test new block positions.";break;
            case '3':if (!valid) { message = "Calibrate before Planning Preview.";break; }test = true;planning = true;freeze = false;planStable = 0;planAnchor.clear();planLocked = false;message = "Planning only. A/D choose S1/S2/S3, P lock, U unlock, L save CSV.";break;
            case 'a':if (planning) { pattern = pattern == 1 ? 3 : pattern - 1;planLocked = false; }break;
            case 'd':if (planning) { pattern = pattern == 3 ? 1 : pattern + 1;planLocked = false; }break;
            case 'p':if (planning) { auto p = currentPlan();if (p.valid) { lockedPlan = p;planLocked = true;message = "Preview locked. Scene changes automatically unlock it. No commands sent."; } else message = p.reason; }break;
            case 'u':planLocked = false;message = "Preview unlocked.";break;
            case 'l':savePlan();break;
            case '-':case '+':case '=':
                noiseLimitPx = std::max(1.0, std::min(8.0, noiseLimitPx + (k == '-' ? -.5 : .5)));
                valid = false;gridQualityReady = false;stable = 0;previous.clear();pointHistory.clear();lastRawCentres.clear();
                message = "Noise limit " + fmt(noiseLimitPx, 1) + "px: collecting new frames; fit/physical validation still required";break;
            case 'x':if (comparisonView) { comparisonView = false;freeze = false;message = "Returned to live coordinate view"; }
                    else compareNine();break;
            case 't':rawView = !rawView;message = rawView ? "Raw live image: overlays hidden (no stabilisation/warping)" : "Median markers from fresh frames; motion gate checks raw spread and drift";break;
            case 'c':calibration();break;case 'i':invalidate("Mapping invalidated. Recalibrate after camera or fixture movement.");break;
            case 'v':labelsOK = !labelsOK;if (!labelsOK)valid = false;message = "P1-P9 direction confirmation changed.";break;case 'w':refsOK = !refsOK;if (!refsOK)valid = false;message = "Confirm ONLY after robot reference XY has been physically checked.";break;
            case 'h':planeOK = !planeOK;if (!planeOK)valid = false;message = "Confirm same ~36 mm block-top plane and unchanged support height.";break;
            case 'f':if (!test) { flipRows = !flipRows;invalidate("Row order reversed. Recheck P1-P9."); }break;
            case 'j':if (!test) { flipCols = !flipCols;invalidate("Column order reversed. Recheck P1-P9."); }break;
            case 'r':sample = 0;message = "Click RED top face to sample hue.";break;case 'g':sample = 1;message = "Click GREEN top face to sample hue.";break;case 'b':sample = 2;message = "Click BLUE top face to sample hue.";break;
            case 'k':cfg = ColourModel{};invalidate("Default colour thresholds restored.");break;
            case 's':snapshot();break;case 'e':beginMeasurement();break;case ' ':freeze = !freeze;message = freeze ? "Frame frozen; do not calibrate a moved camera from stale imagery." : "Live frames resumed.";break;
            case 'n':if (opt.source == "demo") { block = { 55. + ((++trial * 47) % 115),55. + ((trial * 31) % 110) }; }break;
            case 'm':if (opt.source == "demo") { pose = (pose + 1) % 3;invalidate("Synthetic camera changed. Return to Calibration and fit again."); }break;
            case 'o':if (opt.source == "demo") { missing = !missing;invalidate("Synthetic P5 visibility changed."); }break;
            case 'q':exit = true;break;
            }
        }
        void click(int x, int y) {
            if (entering)return;
            if (test && !planning && cv::Rect(865, 620, 380, 32).contains({ x,y })) { action('x');return; }
            if (!test && !planning && cv::Rect(1110, 590, 50, 32).contains({ x,y })) { action('-');return; }
            if (!test && !planning && cv::Rect(1170, 590, 50, 32).contains({ x,y })) { action('+');return; }for (auto& b : buttons)if (b.r.contains({ x,y })) { action(b.action);return; }
            if (x >= 855 && y >= 350 && y < 435) { char key = y < 385 ? 'v' : y < 410 ? 'w' : 'h';action(key);return; }
            if (planning && x >= 865 && x < 1255 && y >= 430 && y < 464) { pattern = std::min(3, 1 + (x - 865) / 130);planLocked = false;return; }
            if (!VIDEO.contains({ x,y }))return;double scale = std::min(double(VIDEO.width) / frame.cols, double(VIDEO.height) / frame.rows);int ox = VIDEO.x + int((VIDEO.width - frame.cols * scale) / 2), oy = VIDEO.y + int((VIDEO.height - frame.rows * scale) / 2);cv::Point p(int((x - ox) / scale), int((y - oy) / scale));if (p.x < 0 || p.y < 0 || p.x >= frame.cols || p.y >= frame.rows)return;
            if (sample >= 0) { sampleAt(p);return; }if (test && !detections.empty()) { double best = 1e12;for (int i = 0;i<int(detections.size());++i) { double d = cv::norm(detections[i].centre - Point2d(p));if (d < best) { best = d;selected = i; } }testStable = 0; }
        }
        void key(int key) { if (key < 0)return;key &= 255;if (entering) { if (key == 27) { entering = false;freeze = false;message = "Measurement cancelled."; } else if (key == 13 || key == 10)finishMeasurement();else if (key == 8) { if (!input.empty())input.pop_back(); } else if ((key >= '0' && key <= '9') || key == '-' || key == '+' || key == '.' || key == ',' || key == ' ') { if (input.size() < 50)input += char(key); }return; }if (key == 27) { exit = true;return; }action(char(std::tolower(key))); }
        void draw() {
            ui = cv::Mat(UIH, UIW, CV_8UC3, BG);auto text = [&](int x, int y, const std::string& s, double scale = .48, cv::Scalar color = FG) {cv::putText(ui, s, { x,y }, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);};
            text(18, 27, "HAND-EYE / CALIBRATION + PLANNING", .64);text(760, 26, sourceName() + " | NO SERIAL / NO ROBOT MOTION", .47, GOLD);
            for (auto& b : buttons) { cv::rectangle(ui, b.r, ((planning && b.action == '3') || (test && !planning && b.action == '2') || (!test && b.action == '1')) ? cv::Scalar(100, 75, 45) : PANEL, cv::FILLED);text(b.r.x + 8, b.r.y + 23, b.label, .46); }
            cv::Mat annotated = frame.clone();auto source = test ? detections : (grid.empty() ? detections : grid);
            auto means = averagedCentres();
            for (int i = 0;i<int(source.size());++i) {
                auto d = source[i];
                if (!test && grid.size() == 9 && means.size() == 9)d.centre = means[i];cv::Scalar color = d.colour == 0 ? cv::Scalar(65, 85, 230) : d.colour == 1 ? cv::Scalar(70, 210, 90) : cv::Scalar(230, 150, 65);cv::polylines(annotated, d.quad, true, color, 2);cv::drawMarker(annotated, d.centre, color, cv::MARKER_CROSS, 14, 2);std::string tag = test ? "B" + std::to_string(i + 1) : grid.empty() ? "?" : "P" + std::to_string(i + 1);
                if (test && valid) { auto p = project(mapping.h, d.centre);tag += " X" + fmt(p.x, 1) + " Y" + fmt(p.y, 1);if (nearBoundary(mapping, p))tag += " EDGE";else if (!inRegion(mapping, p))tag += " OUTSIDE"; }cv::putText(annotated, tag, { d.box.x,std::max(16,d.box.y - 7) }, cv::FONT_HERSHEY_SIMPLEX, .45, color, 1, cv::LINE_AA);
                if (test && i == selected)cv::rectangle(annotated, d.box, cv::Scalar(0, 220, 255), 2);
            }
            if (valid) { std::vector<cv::Point> poly;for (int i : {0, 2, 8, 6}) { auto p = project(mapping.h.inv(), mapping.robot[i]);poly.emplace_back(cvRound(p.x), cvRound(p.y)); }cv::polylines(annotated, poly, true, cv::Scalar(0, 220, 255), 2); }
            else if (!test && grid.size() == 9) { std::vector<cv::Point> poly;for (int i : {0, 2, 8, 6})poly.emplace_back(cvRound(grid[i].centre.x), cvRound(grid[i].centre.y));cv::polylines(annotated, poly, true, cv::Scalar(120, 170, 180), 1); }
            if (rawView)annotated = frame.clone();
            double scale = std::min(double(VIDEO.width) / frame.cols, double(VIDEO.height) / frame.rows);cv::Mat resized;cv::resize(annotated, resized, {}, scale, scale);int ox = VIDEO.x + (VIDEO.width - resized.cols) / 2, oy = VIDEO.y + (VIDEO.height - resized.rows) / 2;resized.copyTo(ui(cv::Rect(ox, oy, resized.cols, resized.rows)));
            cv::rectangle(ui, { 850,100,415,615 }, PANEL, cv::FILLED);int y = 125;auto row = [&](std::string s, cv::Scalar col = FG) {text(865, y, s, .46, col);y += 25;};
            row(planning ? "PLANNING PREVIEW" : test ? "COORDINATE TEST" : "CALIBRATION", GOLD);row("Tops: " + std::to_string(detections.size()) + "  R:" + std::to_string(std::count_if(detections.begin(), detections.end(), [](const Detection& d) {return d.colour == 0;})) + " G:" + std::to_string(std::count_if(detections.begin(), detections.end(), [](const Detection& d) {return d.colour == 1;})) + " B:" + std::to_string(std::count_if(detections.begin(), detections.end(), [](const Detection& d) {return d.colour == 2;})));row(planning ? "Stable scene: " + std::to_string(std::min(planStable, STABLE_FRAMES)) + " / 20 frames" : "Stable grid: " + std::to_string(std::min(stable, STABLE_FRAMES)) + " / 20 frames");row("Source resolution: " + std::to_string(frame.cols) + " x " + std::to_string(frame.rows));row(valid ? "Mapping: SAVED (not field validated)" : gridQualityReady ? "Mapping: READY TO FIT" : "Mapping: NOT READY");
            if (valid) { row("Fit RMS " + fmt(mapping.rms) + " / max " + fmt(mapping.max) + " mm"); }
            else row("Fit gate: max " + fmt(opt.fitLimit) + " mm (setup only)");
            row("Lens distortion correction: NONE", GOLD);row("Centre region: 150 x 140 mm nominal");
            text(865, 330, "Raw step " + fmt(rawPointStep, 1) + " px | drift " + fmt(rawPointDrift, 1) + " px", .42, GOLD);
            text(865, 345, "Spread " + fmt(stableSpread, 1) + " / drift " + fmt(windowDrift, 1) + " px", .36, MUTED);
            y = 370;row(std::string(labelsOK ? "[x] " : "[ ] ") + "V  P1-P9 labels/direction checked");row(std::string(refsOK ? "[x] " : "[ ] ") + "W  Robot reference XY verified");row(std::string(planeOK ? "[x] " : "[ ] ") + "H  Same ~36 mm block-top plane");
            y = 445;
            if (planning) {
                for (int s = 1;s <= 3;++s) { cv::Rect r(865 + (s - 1) * 130, 430, 120, 34);cv::rectangle(ui, r, s == pattern ? cv::Scalar(100, 75, 45) : BG, cv::FILLED);text(r.x + 30, 453, "S" + std::to_string(s), .52); }
                y = 487;Plan p = currentPlan();row("Target S" + std::to_string(pattern) + " (fixed stack)", GOLD);
                for (int l = 0;l < 3;++l) { int color = PATTERNS[pattern - 1][2 - l];row("L" + std::to_string(l + 1) + " " + COLOUR_NAMES[color] + (l == 0 ? " (bottom)" : l == 2 ? " (top)" : ""));if (p.valid)row(" B" + std::to_string(p.source[l] + 1) + " X" + fmt(p.xy[l].x, 1) + " Y" + fmt(p.xy[l].y, 1), MUTED); }
                row(p.valid ? (planLocked ? "LOCKED / PREVIEW ONLY" : "LIVE / PREVIEW ONLY") : p.reason, GOLD);
                text(865, 701, "A/D pattern | P lock | U unlock | L save", .40, MUTED);
            }
            else if (!test) {
                row("F reverse rows / J reverse columns");row("Distance checks (mapped vs reference):");
                if (valid && grid.size() == 9) { for (auto pair : std::vector<std::pair<int, int>>{ {0,1},{0,3},{6,8} }) { double px = cv::norm(grid[pair.first].centre - grid[pair.second].centre), nom = cv::norm(refs[pair.first] - refs[pair.second]);double mapped = cv::norm(project(mapping.h, grid[pair.first].centre) - project(mapping.h, grid[pair.second].centre));row("P" + std::to_string(pair.first + 1) + "-P" + std::to_string(pair.second + 1) + ": " + fmt(px, 0) + " px / " + fmt(mapped, 1) + " mm");row("  reference " + fmt(nom, 1) + " mm", MUTED); } }
                else row("Fit after confirming all three checks.");
            }
            else if (valid && !detections.empty()) { auto p = project(mapping.h, detections[selected].centre);row("Selected B" + std::to_string(selected + 1) + ": X " + fmt(p.x) + " Y " + fmt(p.y));row(regionLabel(mapping, p), GOLD);row("To centre boundary: " + fmt(regionDistance(mapping, p), 2) + " mm");row(selectStable ? "Selected block stable" : "Waiting for selected block to settle");row("B numbers are detections, not P slots.");row("E: enter measured robot X,Y.");row("Fit error is not independent accuracy."); }
            if (test && !planning) { cv::Rect r(865, 620, 380, 32);cv::rectangle(ui, r, BG, cv::FILLED);text(875, 642, comparisonView ? "X Back to live camera" : "X 9-point comparison (grid only)", .43, FG); }
            if (comparisonView && comparison.size() == 9) {
                cv::rectangle(ui, VIDEO, BG, cv::FILLED);
                text(30, 135, "P1-P9 COORDINATE COMPARISON / mm", .60, FG);
                text(30, 166, "Original nine-block grid required. Snapshot is frozen.", .43, MUTED);
                text(30, 207, "Point    Code X     Code Y    Camera X   Camera Y     dX       dY      XY diff", .42, GOLD);
                double square = 0, maximum = 0;
                for (int i = 0;i < 9;++i) {
                    auto r = comparison[i];auto d = r.predicted - r.reference;int y = 243 + i * 39;
                    text(30, y, "P" + std::to_string(i + 1), .48, FG);
                    const double vals[] = { r.reference.x,r.reference.y,r.predicted.x,r.predicted.y,d.x,d.y,r.error };
                    const int cols[] = { 110,210,310,410,520,615,715 };
                    for (int j = 0;j < 7;++j)text(cols[j], y, fmt(vals[j], 2), .43, j >= 4 ? GOLD : FG);
                    square += r.error * r.error;maximum = std::max(maximum, r.error);
                }
                text(30, 625, "RMS " + fmt(std::sqrt(square / 9), 2) + " mm | max " + fmt(maximum, 2) + " mm", .50, FG);
                text(30, 660, "Difference = sqrt(dX*dX + dY*dY). X returns live.", .43, MUTED);
                text(30, 690, "Not an independent physical accuracy test.", .43, GOLD);
            }
            if (!planning && !test) {
                text(865, 612, "Noise limit: " + fmt(noiseLimitPx, 1) + " px", .42, FG);
                for (int i = 0;i < 2;++i) { cv::Rect r(1110 + i * 60, 590, 50, 32);cv::rectangle(ui, r, BG, cv::FILLED);text(r.x + 19, r.y + 23, i ? "+" : "-", .55, FG); }
                text(865, 640, gridQualityReady ? "READY: verify V/W/H, then Fit + save" : stabilityReason, .37, gridQualityReady ? cv::Scalar(70, 210, 90) : GOLD);
            }
            if (!planning)text(865, 665, rawView ? "T: show filtered markers" : "T: raw view / hide overlays", .40, MUTED);
            if (!planning) { y = 690;text(865, y, freeze ? "FRAME FROZEN" : "SPACE: freeze / resume", .45, freeze ? GOLD : MUTED); }
            text(18, 744, "R/G/B then click top face: sample colour | K: default colours | I: invalidate after camera movement", .45);
            text(18, 771, "P1-P9 must be checked. Nine-point symmetry cannot identify direction automatically.", .46, GOLD);
            text(18, 797, "Frame limits block CENTRES only. Jaw clearance, Z, rotations, reachability and physical accuracy are not validated.", .43, MUTED);
            std::string msg = entering ? "ACTUAL ABB X,Y: " + input + "_" : message;int yy = 829;while (!msg.empty() && yy < 895) { size_t cut = std::min<size_t>(125, msg.size());if (cut < msg.size()) { auto pos = msg.rfind(' ', cut);if (pos != std::string::npos && pos > 30)cut = pos; }text(18, yy, msg.substr(0, cut), .46, GOLD);msg.erase(0, cut);yy += 23; }
        }
    };
    H syntheticCamera(int pose) { if (pose == 1)return H(2.5, .25, 110, -.04, -2.2, 610, .00075, -.0002, 1);if (pose == 2)return H(2.8, -.1, 90, .1, -2.4, 590, .0002, .0005, 1);return H(2.8, .12, 70, .06, -2.5, 620, .00055, .00025, 1); }
    cv::Mat synthetic(const App& a) { cv::Mat image(720, 960, CV_8UC3, cv::Scalar(205, 205, 205));auto draw = [&](Point2d p, int color) {std::vector<cv::Point> q;for (auto delta : std::vector<Point2d>{ {-18,-18},{18,-18},{18,18},{-18,18} }) { auto v = project(syntheticCamera(a.pose), p + delta);q.emplace_back(cvRound(v.x), cvRound(v.y)); }cv::Scalar colors[] = { cv::Scalar(45,45,220),cv::Scalar(45,180,45),cv::Scalar(210,70,40) };cv::fillConvexPoly(image, q, colors[color % 3], cv::LINE_8);};if (a.planning) { draw(a.block, 0);draw({ 165,150 }, 1);draw({ 65,65 }, 2); } else if (a.test)draw(a.block, 0);else { auto p = localGrid();for (int i = 0;i < 9;++i)if (!a.missing || i != 4)draw(p[i], i % 3); }return image; }
    bool loadReferences(const std::string& filename, std::vector<Point2d>& refs) {
        std::ifstream f(filename);if (!f)return false;std::string line;std::getline(f, line);std::vector<Point2d> points(9);std::array<bool, 9> seen{};
        while (std::getline(f, line)) { if (line.empty())continue;std::replace(line.begin(), line.end(), ',', ' ');std::istringstream s(line);std::string name, extra;double x, y;if (!(s >> name >> x >> y) || (s >> extra) || name.size() != 2 || name[0] != 'P' || name[1] < '1' || name[1]>'9' || !std::isfinite(x) || !std::isfinite(y))return false;int i = name[1] - '1';if (seen[i])return false;seen[i] = true;points[i] = { x,y }; }
        if (!std::all_of(seen.begin(), seen.end(), [](bool a) {return a;}) || polygonArea(points) < 100)return false;refs = points;return true;
    }
    bool selfTest() {
        {
            cv::Mat dark(240, 300, CV_8UC3, cv::Scalar(130, 130, 130));
            cv::rectangle(dark, { 35,45,40,40 }, cv::Scalar(45, 12, 12), cv::FILLED);
            cv::rectangle(dark, { 110,45,40,40 }, cv::Scalar(12, 45, 12), cv::FILLED);
            cv::rectangle(dark, { 185,45,40,40 }, cv::Scalar(12, 12, 45), cv::FILLED);
            cv::rectangle(dark, { 34,59,4,4 }, cv::Scalar(130, 130, 130), cv::FILLED);
            auto found = detect(dark, ColourModel{});
            if (found.size() != 3)throw std::runtime_error("Dim/notched block detection failed");
            // Rounded corners: full square synthetic fixtures previously missed this regression.
            cv::Mat rounded(220, 300, CV_8UC3, cv::Scalar(120, 120, 120));
            for (int k = 0;k < 3;++k) {
                cv::Point c(50 + k * 90, 100);cv::Scalar colour = k == 0 ? cv::Scalar(0, 0, 160) : k == 1 ? cv::Scalar(0, 160, 0) : cv::Scalar(160, 0, 0);
                cv::rectangle(rounded, cv::Rect(c.x - 15, c.y - 20, 30, 40), colour, cv::FILLED);
                cv::rectangle(rounded, cv::Rect(c.x - 20, c.y - 15, 40, 30), colour, cv::FILLED);
                for (int dx : {-15, 15})for (int dy : {-15, 15})cv::circle(rounded, c + cv::Point(dx, dy), 5, colour, cv::FILLED);
            }
            if (detect(rounded, ColourModel{}).size() != 3)throw std::runtime_error("Rounded block corners rejected");
            cv::Mat circle(200, 200, CV_8UC3, cv::Scalar(120, 120, 120));cv::circle(circle, { 100,100 }, 30, { 0,0,160 }, cv::FILLED);
            if (!detect(circle, ColourModel{}).empty())throw std::runtime_error("Circular object accepted as square top");
            std::vector<Detection> missing = found;
            if (gridOrder(missing, false, false))throw std::runtime_error("Partial dim grid accepted");
        }
        double worst = 0;for (int pose = 0;pose < 3;++pose) {
            App a;a.opt.source = "demo";a.pose = pose;for (int i = 0;i < STABLE_FRAMES + 1;++i) { a.frame = synthetic(a);a.update(); }if (a.grid.size() != 9 || a.stable < STABLE_FRAMES)throw std::runtime_error("Nine-point automatic detection/stability failed");Mapping m;if (!makeMapping(a.grid, a.refs, a.frame.size(), m) || m.max > 2)throw std::runtime_error("Fit failed");a.mapping = m;a.valid = true;a.test = true;
            for (auto p : std::vector<Point2d>{ {55,55},{100,93},{178,160},{125,145},{180,50} }) { a.block = p;a.frame = synthetic(a);a.update();if (a.detections.size() != 1)throw std::runtime_error("Independent block detection failed");auto pred = project(m.h, a.detections[0].centre);Point2d actual{ 552.65 - p.y,p.x - 122.8 };double e = cv::norm(pred - actual);worst = std::max(worst, e);if (e > 2 || !inRegion(m, pred))throw std::runtime_error("Unseen-position mapping failed"); }
            a.invalidate("test");if (a.valid || a.labelsOK || a.refsOK || a.planeOK)throw std::runtime_error("Invalidation failed");
        }
        App a;a.opt.source = "demo";a.missing = true;a.frame = synthetic(a);a.update();if (!a.grid.empty())throw std::runtime_error("Missing point accepted");H h;if (fitH({ {0,0},{1,1},{2,2},{3,3} }, { {0,0},{1,1},{2,2},{3,3} }, h))throw std::runtime_error("Degenerate fit accepted");
        a.missing = false;a.frame = synthetic(a);a.update();Mapping m;if (!makeMapping(a.grid, a.refs, a.frame.size(), m))throw std::runtime_error("Region test mapping failed");if (inRegion(m, { 600,200 }))throw std::runtime_error("Outside region accepted");
        // Verify smoothing never hides motion or fills missing detections.
        {
            App jitter;jitter.opt.source = "demo";cv::Mat base = synthetic(jitter);
            for (int i = 0;i < 30;++i) { cv::Matx23d shift(1, 0, i % 2, 0, 1, 0);cv::warpAffine(base, jitter.frame, shift, base.size());jitter.update(); }
            if (jitter.stable < 20 || jitter.pointHistory.size() != 20)throw std::runtime_error("Small raw jitter did not stabilise");
            cv::Matx23d shift(1, 0, 16, 0, 1, 0);cv::warpAffine(base, jitter.frame, shift, base.size());jitter.update();
            if (jitter.stable != 1 || jitter.pointHistory.size() != 1)throw std::runtime_error("Smoothing concealed actual movement");
            jitter.missing = true;jitter.frame = synthetic(jitter);jitter.update();
            if (jitter.stable || !jitter.pointHistory.empty())throw std::runtime_error("Stale points survived missing block");
        }
        {
            App measured;std::vector<Detection> fixture;
            for (int i = 0;i < 9;++i) { Detection d;d.centre = { double(60 + i % 3 * 80),double(60 + i / 3 * 80) };d.area = 900;d.colour = i % 3;fixture.push_back(d); }
            for (int t = 0;t < 30;++t) { measured.grid = fixture;for (auto& d : measured.grid)d.centre.x += (t % 2 ? 1.8 : -1.8);measured.updateGridStability(); }
            if (measured.stable != 20 || !measured.gridQualityReady)throw std::runtime_error("Bounded segmentation jitter rejected");
            measured.invalidate("test");
            for (int t = 0;t < 20;++t) { measured.grid = fixture;for (auto& d : measured.grid)d.centre.x += t * .4;measured.updateGridStability(); }
            if (measured.gridQualityReady)throw std::runtime_error("Slow physical drift accepted");
            measured.grid = fixture;measured.grid.pop_back();measured.updateGridStability();
            if (measured.stable || !measured.pointHistory.empty())throw std::runtime_error("Missing grid retained stale samples");
        }
        {
            App noisy;std::vector<Detection> fixture;
            for (int i = 0;i < 9;++i) { Detection d;d.centre = { double(60 + i % 3 * 80),double(60 + i / 3 * 80) };d.area = 900;d.colour = i % 3;fixture.push_back(d); }
            auto run = [&]() {for (int t = 0;t < 30;++t) { noisy.grid = fixture;for (auto& d : noisy.grid)d.centre.x += (t % 2 ? 1.8 : -1.8);noisy.updateGridStability(); }};
            noisy.noiseLimitPx = 1.5;run();
            if (noisy.stable != 20 || noisy.gridQualityReady)throw std::runtime_error("Noise rejection count/display incorrect");
            noisy.action('+');run();if (!noisy.gridQualityReady)throw std::runtime_error("Adjusted noise limit not applied");
            noisy.labelsOK = noisy.refsOK = noisy.planeOK = true;noisy.noiseLimitPx = 1.0;noisy.gridQualityReady = false;noisy.calibration();
            if (noisy.valid || noisy.message.find("noise spread") == std::string::npos)throw std::runtime_error("Fit failed to explain quality rejection");
        }
        {
            Mapping boundary;boundary.robot = nominalRobot();
            if (!nearBoundary(boundary, { 372.34,68.30 }) || inRegion(boundary, { 372.34,68.30 }))throw std::runtime_error("Edge display expanded allowed region");
            if (regionLabel(boundary, { 442.65,-7.8 }) != "INSIDE" || regionLabel(boundary, { 600,200 }) != "OUTSIDE")throw std::runtime_error("Boundary classifications wrong");
        }
        // Exercise operator gates, real save/readback and independent-check recording.
        a = App{};a.opt.source = "demo";for (int i = 0;i < 25;++i) { a.frame = synthetic(a);a.update(); }
        a.calibration();if (a.valid)throw std::runtime_error("Operator confirmation gate bypassed");
        const auto original = fileutil::currentPath();fileutil::createDirectory("qa_outputs");fileutil::currentPath("qa_outputs");
        try {
            a.labelsOK = a.refsOK = a.planeOK = true;a.calibration();if (!a.valid)throw std::runtime_error("Calibration save failed");
            cv::FileStorage f((a.session + "/" + "calibration.yml"), cv::FileStorage::READ);cv::Mat stored;f["H_pixel_to_robot_xy"] >> stored;int physical = 1, sim = 0;f["physically_validated"] >> physical;f["simulation"] >> sim;
            if (physical != 0 || sim != 1 || cv::norm(stored - cv::Mat(a.mapping.h)) > 1e-8)throw std::runtime_error("Saved calibration metadata or matrix mismatch");
            a.test = true;a.compareNine();
            if (!a.comparisonView || !a.freeze || a.comparison.size() != 9 || !fileutil::exists(a.comparisonPath))throw std::runtime_error("Nine-point comparison export failed");
            for (int i = 0;i < 9;++i)if (a.comparison[i].error > 2 || cv::norm(a.comparison[i].reference - a.refs[i]) > 1e-9)throw std::runtime_error("Comparison point identity mismatch");
            a.action('x');if (a.freeze || a.comparisonView)throw std::runtime_error("Comparison did not return live");
            a.test = true;for (int i = 0;i < 15;++i) { a.frame = synthetic(a);a.update(); }a.beginMeasurement();if (!a.entering)throw std::runtime_error("Validation entry failed");
            a.input = "invalid";a.finishMeasurement();if (!a.entering)throw std::runtime_error("Invalid actual XY accepted");a.input = "427.65,-17.8";a.finishMeasurement();
            if (a.entering || !fileutil::exists(a.session + "/" + "independent_checks.csv"))throw std::runtime_error("Validation log failed");
            a.action('3');for (int i = 0;i < 25;++i) { a.frame = synthetic(a);a.update(); }
            for (int pattern = 1;pattern <= 3;++pattern) { a.pattern = pattern;auto p = a.currentPlan();if (!p.valid)throw std::runtime_error("Three-colour planning failed");for (int l = 0;l < 3;++l) { if (p.colours[l] != PATTERNS[pattern - 1][2 - l])throw std::runtime_error("Pattern order changed");for (int j = 0;j < l;++j)if (p.source[l] == p.source[j])throw std::runtime_error("Duplicate source in plan"); } }
            a.action('p');if (!a.planLocked)throw std::runtime_error("Stable plan lock failed");a.savePlan();
            a.block.x += 8;a.frame = synthetic(a);a.update();if (a.planLocked || a.currentPlan().valid)throw std::runtime_error("Moved scene did not invalidate preview");
            auto absent = a.detections;absent.erase(std::remove_if(absent.begin(), absent.end(), [](auto& d) {return d.colour == 1;}), absent.end());if (makePlan(absent, a.mapping, 1).valid)throw std::runtime_error("Missing colour accepted");
            auto outside = a.detections;for (auto& d : outside)if (d.colour == 1)d.centre = project(a.mapping.h.inv(), { 600,200 });if (makePlan(outside, a.mapping, 1).valid)throw std::runtime_error("Outside block planned");
            a.action('v');if (a.valid)throw std::runtime_error("Unconfirm did not invalidate");
            fileutil::currentPath(original);
        }
        catch (...) { fileutil::currentPath(original);throw; }
        std::cout << "PASS: automatic nine-top detection in 3 camera poses; 15 unseen points; worst synthetic XY error " << worst << " mm; operator gates, save/readback, validation logging, all 3 planning patterns, plan save, movement unlock, missing colour and outside-source rejection.\n";return true;
    }
    void mouse(int event, int x, int y, int, void* ptr) { if (event == cv::EVENT_LBUTTONDOWN) { auto& a = *static_cast<App*>(ptr);try { a.click(x, y); } catch (const std::exception& e) { a.message = e.what(); } } }
} // namespace calibrationui

constexpr int N = 9, CAM = 0, W = 1280, H = 1120, CAMW = 850, CAMH = 620;
constexpr int STABLE = 6, LOST_FRAMES = 20, ROI = 40;
constexpr double MIN_AREA = 350;
const string CONFIG = "handeye_config.yml";
enum Colour { UNKNOWN = 0, RED = 1, GREEN = 2, BLUE = 3 };

const char* cname(int c) {
    switch (c) {
    case RED: return "RED";
    case GREEN: return "GREEN";
    case BLUE: return "BLUE";
    default: return "UNKNOWN";
    }
}
Scalar ink(int c) {
    switch (c) {
    case RED: return { 65,85,235 };
    case GREEN: return { 90,210,85 };
    case BLUE: return { 230,155,65 };
    default: return { 135,145,160 };
    }
}
const Scalar BG(24, 27, 35), PANEL(37, 42, 53), WHITE(235, 240, 245), MUTED(155, 167, 180), GOLD(55, 190, 245);

struct Position { int candidate = 0, streak = 0, confirmed = 0, missing = 0; bool uncertain = true; };
struct Detection {
    Point centre; int colour; Rect box; double area;
    RotatedRect face; double fillRatio = 0; bool measurable = false;
};
struct BlockSample { Point centre; int colour; RotatedRect face; double area; };
constexpr double BLOCK_MM = 40.0;
constexpr int SIZE_SAMPLES = 30;
struct Plan { bool valid = false; array<int, 3> source{ {-1,-1,-1} }; array<int, 3> colours{ {0,0,0} }; string message; };

struct App {
    array<Point, N> points{}; int mapped = 0; bool mapping = false;
    array<Position, N> positions{};
    int selected = 1; bool autoMode = true;
    bool busy = false;
    bool autoProposal = true;
    bool learnPick = false, learning = false;
    Point learnCentre{}; int learnColour = UNKNOWN;
    vector<BlockSample> sizeSamples;
    Size2f learnedFace{}; Size learnedResolution{};
    string sizeCapturePath;
    vector<Detection> lastDetections; Mat lastFrame;
    bool dynamicTestMode = false, dynamicSending = false;
    int dynamicStep = 0, dynamicToken = 0, dynamicColour = -1;
    chrono::steady_clock::time_point hoverSentAt{};
    Point2d dynamicXY{};
    Point2d dynamicTrim{};
    string dynamicStatus = "Supervised test off";
    int dynamicRequest = 0;
    bool calibrationMode = false; calibrationui::App* calibration = nullptr;
    bool setupOpen = false; int pendingKey = -1; int completedCycles = 0;
    array<array<int, 3>, 3> sentColours{};
    array<chrono::steady_clock::time_point, 3> emptySince{}, retryAt{}, restoredSince{};
    array<bool, 3> returnStarted{};
    array<chrono::steady_clock::time_point, 3> stackSentAt{};
    bool sampleMode = false; int sampleColour = RED;
    array<int, 4> hue{ {0,0,60,110} };
    array<int, 4> tolerance{ {0,12,15,15} };
    int satMin = 65, valMin = 60;
    string message = "Load saved P1-P9 reference or calibrate once using M (9 blocks for auto proposal)";
    bool serialConnected = false; sp_port* port = nullptr; bool planReceiverReady = false;
    int lastSent = -1; string comName = "NONE"; bool commandLatched = false;
    bool planLocked = false; array<Plan, 3> queued{};
    bool allowRepeatPickups = false; // default: unused blocks only
    array<bool, N> processedBlocks{}; // handled blocks tracked at their current return slots
    array<bool, N> reservedSources{};            // pickup positions used by a sent stack, freed by T
    chrono::steady_clock::time_point lastPlanRefresh{};
    bool planRefreshReady = false;
    array<bool, 3> sentPlans{ {false,false,false} };   // TX only, NOT ABB acknowledgement
    array<array<int, 3>, 3> sentSources{ {{{-1,-1,-1}},{{-1,-1,-1}},{{-1,-1,-1}}} };
    array<int, 3> sendOrder{ {0,0,0} }; int sentCount = 0;
    // Random return plan (per stack): destination P index for level 1,2,3 (0-based).
    array<array<int, 3>, 3> returnDest{ {{{-1,-1,-1}},{{-1,-1,-1}},{{-1,-1,-1}}} };
    array<bool, 3> returnSent{ {false,false,false} };
    array<bool, N> returnReserved{};             // destinations promised to a pending return
    bool releaseArmed = false;                   // second T forces release without a return plan
    mt19937 rng{ random_device{}() };
    bool autoCandidate = false; array<Point, N> candidatePoints{}; int candidateStreak = 0;
    bool verifiedMap = false;
    bool showTargetGrid = true;
    int demoStep = 0;
    int zoneSetup = 0; array<Point, 3> stackZones{}; array<bool, 3> zoneDefined{ {false,false,false} };
    array<int, 3> stackTop{ {0,0,0} }, topCandidate{ {0,0,0} }, topStreak{ {0,0,0} };
    array<vector<int>, 3> observedHistory{};
    Mat lastHSV; Size cameraSize{}; Rect imageRect{ 15,100,CAMW,CAMH };
};

// Small directory helpers keep this source compatible with Visual Studio's
// default C++14 mode; no std::filesystem or extra library is needed.
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
    bool exists(const string& path) { ifstream file(path, ios::binary); return file.good(); }
}

// ---------------------------------------------------------------- drawing helpers
void label(Mat& m, const string& s, Point p, double size = .56, Scalar c = WHITE, int thick = 1) {
    putText(m, s, p, FONT_HERSHEY_SIMPLEX, size, c, thick, LINE_AA);
}
void fill(Mat& m, Rect r, Scalar c) { rectangle(m, r, c, FILLED, LINE_AA); }
void panel(Mat& m, Rect r) { fill(m, r, PANEL); rectangle(m, r, Scalar(57, 63, 76), 1); }
void button(Mat& m, Rect r, const string& s, bool selected = false) {
    fill(m, r, selected ? Scalar(66, 93, 116) : Scalar(53, 60, 74));
    rectangle(m, r, selected ? GOLD : Scalar(77, 84, 97), 1);
    double font = .48;
    while (font > .24 && getTextSize(s, FONT_HERSHEY_SIMPLEX, font, 1, nullptr).width > r.width - 16) font -= .02;
    label(m, s, Point(r.x + 8, r.y + r.height / 2 + 5), font, selected ? WHITE : MUTED, 1);
}
// Replaces Sleep(): keeps the OpenCV window responsive and needs no <windows.h>.
static void hold(int ms) {
    const auto until = chrono::steady_clock::now() + chrono::milliseconds(ms);
    while (chrono::steady_clock::now() < until) waitKey(10);
}

void resetObservations(App& a) { a.positions.fill(Position{}); }

// ---------------------------------------------------------------- config
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
    if (!f["autoProposal"].empty()) { int v; f["autoProposal"] >> v; a.autoProposal = v != 0; }
    if (!f["learnedWidthPx"].empty()) {
        f["learnedWidthPx"] >> a.learnedFace.width; f["learnedHeightPx"] >> a.learnedFace.height;
        f["learnedResolution"] >> a.learnedResolution; f["sizeCapturePath"] >> a.sizeCapturePath;
    }
    if (!f["dynamicTrimX"].empty()) {
        f["dynamicTrimX"] >> a.dynamicTrim.x;f["dynamicTrimY"] >> a.dynamicTrim.y;
        if (!isfinite(a.dynamicTrim.x) || !isfinite(a.dynamicTrim.y) || abs(a.dynamicTrim.x) > 5 || abs(a.dynamicTrim.y) > 5)a.dynamicTrim = {};
    }
    if (!f["allowRepeatPickups"].empty()) { int enabled = 0; f["allowRepeatPickups"] >> enabled; a.allowRepeatPickups = enabled != 0; }
    int historyIndex = 0;
    for (auto entry : f["processedBlocks"]) {
        int used = 0; entry >> used;
        if (historyIndex < N) a.processedBlocks[historyIndex++] = used != 0;
    }
    int n = 0; f["mapped"] >> n;
    FileNode p = f["points"]; int i = 0;
    for (auto it = p.begin(); it != p.end() && i < N; ++it) { (*it) >> a.points[i++]; }
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

// ---------------------------------------------------------------- vision
void findColour(const Mat& hsv, Mat& camera, int colour, const App& a, vector<Detection>& detections) {
    Mat mask(hsv.size(), CV_8UC1, Scalar(0)), part;
    int lo = 0, hi = 0, s = a.satMin, v = a.valMin;
    if (colour == RED && a.hue[RED] == 0) {
        inRange(hsv, Scalar(0, 100, 80), Scalar(10, 255, 255), mask);
        inRange(hsv, Scalar(170, 100, 80), Scalar(179, 255, 255), part);
        bitwise_or(mask, part, mask);
    }
    else if (colour == GREEN && a.hue[GREEN] == 60) { inRange(hsv, Scalar(35, 70, 60), Scalar(85, 255, 255), mask); }
    else if (colour == BLUE && a.hue[BLUE] == 110) { inRange(hsv, Scalar(90, 80, 60), Scalar(130, 255, 255), mask); }
    else {
        lo = a.hue[colour] - a.tolerance[colour]; hi = a.hue[colour] + a.tolerance[colour];
        for (int offset = -180; offset <= 180; offset += 180) {
            int l = max(0, lo + offset), r = min(179, hi + offset);
            if (l > r) continue;
            inRange(hsv, Scalar(l, s, v), Scalar(r, 255, 255), part);
            bitwise_or(mask, part, mask);
        }
    }
    morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    vector<vector<Point>> contours; findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        double area = contourArea(contour); if (area < MIN_AREA) continue;
        Moments mo = moments(contour); if (mo.m00 == 0) continue;
        Point p(cvRound(mo.m10 / mo.m00), cvRound(mo.m01 / mo.m00));
        Rect box = boundingRect(contour);
        RotatedRect face = minAreaRect(contour);
        double shortSide = min(face.size.width, face.size.height);
        double longSide = max(face.size.width, face.size.height);
        double ratio = area / max(1.0, double(face.size.area()));
        bool measurable = shortSide >= 12 && longSide / max(1.0, shortSide) <= 1.45 && ratio >= .80 &&
            box.x > 2 && box.y > 2 && box.br().x < hsv.cols - 2 && box.br().y < hsv.rows - 2;
        detections.push_back({ p, colour, box, area, face, ratio, measurable });
        Point2f corners[4]; face.points(corners);
        for (int i = 0; i < 4; ++i) line(camera, corners[i], corners[(i + 1) % 4], ink(colour), 2, LINE_AA);
        string text = string(cname(colour)) + " " + to_string(cvRound(shortSide)) + "x" + to_string(cvRound(longSide)) + "px";
        label(camera, text, Point(box.x, max(17, box.y - 5)), .43, ink(colour), 1);
    }
}

// Reset is persisted; automatic grid proposal stays off until the operator maps again.
void resetSettings(App& a, bool all) {
    if (a.busy || a.sentCount) { a.message = "Reset unavailable during a cycle"; return; }
    if (all) {
        auto calibration = a.calibration;
        auto port = a.port; bool connected = a.serialConnected; string com = a.comName;
        int last = a.lastSent, cycles = a.completedCycles; bool latched = a.commandLatched;
        a = App{}; a.calibration = calibration; a.port = port; a.serialConnected = connected; a.comName = com;
        a.lastSent = last; a.completedCycles = cycles; a.commandLatched = latched;
    }
    a.processedBlocks.fill(false); // mapping IDs are no longer valid after remapping
    a.points = {}; a.mapped = 0; a.mapping = false; a.verifiedMap = false;
    a.zoneDefined.fill(false); a.stackZones = {};
    a.stackTop.fill(0); a.topCandidate.fill(0); a.topStreak.fill(0);
    for (auto& h : a.observedHistory) h.clear();
    a.autoProposal = false; a.autoCandidate = false; a.candidateStreak = 0;
    a.sampleMode = false; a.zoneSetup = 0; a.learnPick = false; a.learning = false; a.sizeSamples.clear();
    a.queued = {}; a.planRefreshReady = false; a.planLocked = false; resetObservations(a);
    a.setupOpen = true;
    if (!saveConfig(a)) { a.message = "Reset applied in memory but settings file cannot be written"; return; }
    a.message = all ? "Settings reset and saved. COM kept open; measurement archives retained." :
        "P1-P9 and stack zones cleared and saved. Click MAP to start again.";
}

float medianValue(vector<float> v) {
    sort(v.begin(), v.end()); return v[v.size() / 2];
}
void saveBlockCapture(App& a) {
    try {
        string id = to_string(chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count());
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
            csv << i << ',' << cname(r.colour) << ',' << r.centre.x << ',' << r.centre.y << ','
                << r.face.size.width << ',' << r.face.size.height << ',' << r.face.angle << ',' << r.area;
            Point2f corners[4]; r.face.points(corners);
            for (const auto& p : corners) csv << ',' << p.x << ',' << p.y;
            csv << ',' << BLOCK_MM << '\n';
        }
        csv.flush(); if (!csv) throw runtime_error("sample write failed");
        Size2f learned(medianValue(widths), medianValue(heights));
        FileStorage meta(folder + "/measurement.yml", FileStorage::WRITE);
        if (!meta.isOpened()) throw runtime_error("cannot write measurement.yml");
        meta << "known_face_mm" << BLOCK_MM << "sample_count" << int(a.sizeSamples.size())
            << "camera_index" << CAM << "frame_size" << a.cameraSize
            << "median_short_side_px" << learned.width << "median_long_side_px" << learned.height
            << "local_mm_per_px_short" << BLOCK_MM / learned.width
            << "local_mm_per_px_long" << BLOCK_MM / learned.height
            << "scope" << "Local apparent face size only; not robot XY calibration. Same camera pose, resolution and block height required.";
        meta.release();
        if (!imwrite(folder + "/camera.png", a.lastFrame)) throw runtime_error("snapshot write failed");
        ofstream visible(folder + "/visible_blocks.csv");
        if (!visible) throw runtime_error("cannot write visible_blocks.csv");
        visible << "colour,centre_u_px,centre_v_px,width_px,height_px,angle_deg,area_px2,measurable\n";
        for (const auto& d : a.lastDetections)
            visible << cname(d.colour) << ',' << d.centre.x << ',' << d.centre.y << ',' << d.face.size.width << ','
            << d.face.size.height << ',' << d.face.angle << ',' << d.area << ',' << d.measurable << '\n';
        visible.flush(); if (!visible) throw runtime_error("visible block export failed");
        a.learnedFace = learned; a.learnedResolution = a.cameraSize; a.sizeCapturePath = folder;
        if (!saveConfig(a)) throw runtime_error("cannot persist reference in handeye_config.yml");
        a.message = "40mm reference saved: " + to_string(cvRound(learned.width)) + "x" + to_string(cvRound(learned.height)) + " camera px; " + folder;
    }
    catch (const exception& e) { a.message = string("Measurement save failed: ") + e.what(); }
    a.learning = false;
}
void updateBlockCapture(App& a, const vector<Detection>& det) {
    if (!a.learning) return;
    const Detection* closest = nullptr; double best = 24.0;
    for (const auto& d : det) if (d.colour == a.learnColour && d.measurable) {
        double distance = norm(d.centre - a.learnCentre);
        if (distance < best) { closest = &d; best = distance; }
    }
    if (!closest) {
        a.sizeSamples.clear(); a.message = "LEARN: block lost / face unclear. Hold still or click Cancel learn."; return;
    }
    if (!a.sizeSamples.empty()) {
        const auto& first = a.sizeSamples.front();
        double w = min(closest->face.size.width, closest->face.size.height);
        double h = max(closest->face.size.width, closest->face.size.height);
        double fw = min(first.face.size.width, first.face.size.height), fh = max(first.face.size.width, first.face.size.height);
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
    vector<Point> p; for (const auto& d : det) p.push_back(d.centre);
    sort(p.begin(), p.end(), [](Point a, Point b) { return a.y < b.y; });
    for (int row = 0; row < 3; ++row) {
        sort(p.begin() + row * 3, p.begin() + row * 3 + 3, [](Point a, Point b) { return a.x < b.x; });
        for (int col = 0; col < 3; ++col) out[row * 3 + col] = p[row * 3 + col];
    }
    for (int i = 0; i < N; ++i) for (int j = i + 1; j < N; ++j)
        if (norm(out[i] - out[j]) < ROI * 1.2) return false;
    double y0 = 0, y1 = 0, y2 = 0;
    for (int j = 0; j < 3; ++j) { y0 += out[j].y; y1 += out[3 + j].y; y2 += out[6 + j].y; }
    return y1 - y0 > ROI * 1.5 && y2 - y1 > ROI * 1.5;
}

array<int, N> matchDetections(const App& a, const vector<Detection>& det) {
    array<int, N> assignment; assignment.fill(-1);
    vector<bool> used(det.size(), false);
    for (int pass = 0; pass < N; ++pass) {
        int bestI = -1, bestJ = -1; double best = double(ROI) * ROI;
        for (int i = 0; i < N; ++i) if (assignment[i] < 0)
            for (int j = 0; j < (int)det.size(); ++j) if (!used[j]) {
                Point delta = det[j].centre - a.points[i]; double d = delta.dot(delta);
                if (d < best) { best = d; bestI = i; bestJ = j; }
            }
        if (bestI < 0) break;
        assignment[bestI] = bestJ; used[bestJ] = true;
    }
    return assignment;
}

// ---------------------------------------------------------------- planning
Plan makePlan(const App& a, int routine) {
    Plan p; // target TOP -> BOTTOM
    const array<array<int, 3>, 3> top{ {{{GREEN,RED,BLUE}},{{RED,GREEN,BLUE}},{{BLUE,GREEN,RED}}} };
    for (int level = 0; level < 3; ++level) {
        int colour = top[routine - 1][2 - level]; p.colours[level] = colour;
        for (int j = 0; j < N; ++j) {
            bool used = false; for (int k = 0; k < level; ++k) if (p.source[k] == j) used = true;
            if (!used && !a.reservedSources[j] && (a.allowRepeatPickups || !a.processedBlocks[j]) && !a.positions[j].uncertain && a.positions[j].confirmed == colour) { p.source[level] = j; break; }
        }
        if (p.source[level] < 0) { p.message = string(a.allowRepeatPickups ? "Need stable " : "Need unused stable ") + cname(colour) + " block"; return p; }
    }
    p.valid = true; p.message = "Ready (vision plan only)"; return p;
}
void refreshPlans(App& a) {
    for (int s = 1; s <= 3; ++s) a.queued[s - 1] = makePlan(a, s);
    a.lastPlanRefresh = chrono::steady_clock::now();
    a.planRefreshReady = true;
}

// ---------------------------------------------------------------- serial
string binary8(int value) { return bitset<8>(static_cast<unsigned long long>(value & 255)).to_string(); }
string byteDescription(int value) {
    ostringstream os;
    os << "DEC " << value << "  HEX 0x" << uppercase << hex << setw(2) << setfill('0') << value
        << "  BIN " << binary8(value);
    return os.str();
}
bool sendCommand(App& a, uint8_t value) {
    if (a.calibrationMode && !a.dynamicSending) { a.message = "Serial commands disabled in calibration/test mode"; return false; }
    if (!a.port || !a.serialConnected) { a.message = "SERIAL OFFLINE: launch with COM port argument"; return false; }
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
void disconnect(App& a) {
    if (a.port) { sp_close(a.port); sp_free_port(a.port); a.port = nullptr; }
    a.serialConnected = false; a.comName = "NONE";
}
void startSerial(App& a, const string& name) {
    sp_port* p = nullptr;
    if (sp_get_port_by_name(name.c_str(), &p) != SP_OK || !p) { a.message = "Serial port not found: " + name; return; }
    if (sp_open(p, SP_MODE_WRITE) != SP_OK) { sp_free_port(p); a.message = "Cannot open serial port: " + name; return; }
    bool ok = sp_set_baudrate(p, 9600) == SP_OK && sp_set_bits(p, 8) == SP_OK &&
        sp_set_parity(p, SP_PARITY_NONE) == SP_OK && sp_set_stopbits(p, 1) == SP_OK &&
        sp_set_flowcontrol(p, SP_FLOWCONTROL_NONE) == SP_OK;
    if (!ok) { sp_close(p); sp_free_port(p); a.message = "Serial configuration failed"; return; }
    a.port = p; a.serialConnected = true; a.comName = name;
    if (!sendCommand(a, 0)) { disconnect(a); return; }
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
// Frame layout: 0, START, 0, f0, 0, f1, ... 0, END, 0
static bool sendFrame(App& a, uint8_t start, const vector<uint8_t>& fields, uint8_t end) {
    if (a.busy) return false;
    struct Guard { bool& b; Guard(bool& v) : b(v) { b = true; } ~Guard() { b = false; } } guard(a.busy);
    if (!sendCommand(a, 0)) return false; hold(800);
    if (!sendCommand(a, start)) return false; hold(800);
    for (uint8_t f : fields) {
        if (!sendCommand(a, 0)) return false; hold(800);
        if (!sendCommand(a, f)) return false; hold(800);
    }
    if (!sendCommand(a, 0)) return false; hold(800);
    if (!sendCommand(a, end)) return false; hold(800);
    if (!sendCommand(a, 0)) return false;
    return true;
}

// STACK PLAN frame (240 ... 241): A1..A3 stack, B1..B9 level 1 pickup, C1..C9 level 2, D1..D9 level 3.
void operatorStart(App& a) {
    if (a.calibrationMode) return;
    if (a.busy) return;
    a.releaseArmed = false;
    if (a.learnPick || a.learning || a.mapping || a.sampleMode || a.zoneSetup) { a.message = "Finish setup before starting"; return; }
    if (!a.serialConnected) { a.message = "START BLOCKED: COM OFFLINE"; return; }
    if (a.commandLatched) { a.message = "START BLOCKED: press C to clear first; robot status UNKNOWN"; return; }
    if (!a.planRefreshReady) { a.message = "START BLOCKED: wait for vision refresh or press P"; return; }
    if (a.sentCount > 0) { a.message = "WAIT: automatic stack/return cycle in progress"; return; }
    if (a.sentCount >= 3) { a.message = "THREE STACKS SENT; return blocks (D) then release (T)"; return; }
    if (a.sentPlans[a.selected - 1]) { a.message = "S" + to_string(a.selected) + " ALREADY SENT; send return (D) then release (T)"; return; }
    if (!a.verifiedMap || a.mapped != N) { a.message = "START BLOCKED: verify P1-P9 map"; return; }
    Plan p = a.queued[a.selected - 1];
    if (!p.valid) { a.message = "START BLOCKED: " + p.message; return; }
    for (int level = 0; level < 3; ++level) {
        const int src = p.source[level];
        if (src < 0 || src >= N || a.reservedSources[src] || (!a.allowRepeatPickups && a.processedBlocks[src])) {
            a.message = "START BLOCKED: source already used or invalid; refresh plan";
            return;
        }
        for (int earlier = 0; earlier < level; ++earlier)
            if (p.source[earlier] == src) { a.message = "START BLOCKED: duplicate pickup in plan"; return; }
    }
    const vector<uint8_t> fields = {
        static_cast<uint8_t>(0xA0 + a.selected),
        static_cast<uint8_t>(0xB0 + p.source[0] + 1),
        static_cast<uint8_t>(0xC0 + p.source[1] + 1),
        static_cast<uint8_t>(0xD0 + p.source[2] + 1)
    };
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
    a.message = "STACK PLAN SENT (NO ACK): S" + to_string(a.selected) +
        " P" + to_string(p.source[0] + 1) + " P" + to_string(p.source[1] + 1) +
        " P" + to_string(p.source[2] + 1) + "; RX expected: 240," + to_string(160 + a.selected) + "," + to_string(177 + p.source[0]) + "," + to_string(193 + p.source[1]) + "," + to_string(209 + p.source[2]) + ",241";
}

// RANDOM RETURN frame (250 ... 251), sent AFTER the pendant shows "1=WAIT FOR RETURN PLAN":
//   0x80+S = stack, 0x10+P = level-3 drop, 0x20+P = level-2 drop, 0x30+P = level-1 drop.
// Destinations are chosen at random among positions that vision currently confirms EMPTY
// and that no other pending return has claimed. The robot unstacks top-first (L3, L2, L1).
void operatorReturn(App& a) {
    if (a.calibrationMode) return;
    if (a.busy) return;
    a.releaseArmed = false;
    if (!a.serialConnected) { a.message = "RETURN BLOCKED: COM OFFLINE"; return; }
    const int s = a.selected - 1;
    if (!a.sentPlans[s]) { a.message = "S" + to_string(a.selected) + " NOT SENT - nothing to return"; return; }
    if (a.returnSent[s]) { a.message = "S" + to_string(a.selected) + " RETURN ALREADY SENT; after robot finishes press T"; return; }
    if (!a.verifiedMap || a.mapped != N) { a.message = "RETURN BLOCKED: verify P1-P9 map"; return; }
    vector<int> freePos;
    for (int i = 0; i < N; ++i)
        if (!a.positions[i].uncertain && a.positions[i].confirmed == UNKNOWN && !a.returnReserved[i])
            freePos.push_back(i);
    if (freePos.size() < 3) {
        a.message = "RETURN BLOCKED: need 3 stable EMPTY positions (have " + to_string(freePos.size()) + ")";
        return;
    }
    shuffle(freePos.begin(), freePos.end(), a.rng);
    const array<int, 3> dest{ {freePos[0], freePos[1], freePos[2]} };   // level 1,2,3 drop positions
    if (a.commandLatched) { operatorClear(a); if (a.commandLatched) return; }
    const vector<uint8_t> fields = {
        static_cast<uint8_t>(0x80 + a.selected),
        static_cast<uint8_t>(0x10 + dest[2] + 1),
        static_cast<uint8_t>(0x20 + dest[1] + 1),
        static_cast<uint8_t>(0x30 + dest[0] + 1)
    };
    if (!sendFrame(a, 250, fields, 251)) return;
    a.commandLatched = true;
    a.returnDest[s] = dest; a.returnSent[s] = true;
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
    if (!a.sentPlans[s]) { a.message = "S" + to_string(a.selected) + " NOT SENT - nothing to release"; return; }
    if (!a.returnSent[s] && !a.releaseArmed) {
        a.releaseArmed = true;
        a.message = "NO RETURN PLAN SENT FOR S" + to_string(a.selected) + ": press T again to FORCE release";
        return;
    }
    // Mark the returned blocks at their NEW locations, not the old pickup IDs.
    // With fewer than nine blocks, an old source can later hold an unused block.
    for (int src : a.sentSources[s]) if (src >= 0 && src < N) {
        a.reservedSources[src] = false;
        a.processedBlocks[src] = false;
    }
    for (int dest : a.returnDest[s]) if (dest >= 0 && dest < N) a.processedBlocks[dest] = true;
    for (int d : a.returnDest[s]) if (d >= 0 && d < N) a.returnReserved[d] = false;
    a.returnDest[s] = { {-1,-1,-1} }; a.returnSent[s] = false;
    a.sentPlans[s] = false;
    ++a.completedCycles;
    a.sentSources[s] = { {-1,-1,-1} };
    array<int, 3> o{ {0,0,0} }; int n = 0;
    for (int j = 0; j < a.sentCount; ++j) if (a.sendOrder[j] != a.selected) o[n++] = a.sendOrder[j];
    a.sendOrder = o; a.sentCount = n;
    a.observedHistory[s].clear(); a.stackTop[s] = 0;
    a.releaseArmed = false;
    resetObservations(a);          // positions become UNCERTAIN until re-confirmed from the camera
    a.planRefreshReady = false;    // fresh plan only after vision has re-read the new layout
    refreshPlans(a);
    bool saved = saveConfig(a);
    a.message = saved ? (a.allowRepeatPickups ? "Cycle complete. Repeat pickups allowed; USED history retained." : "Cycle complete. Returned blocks marked USED; next plan picks unused blocks only.") :
        "Cycle complete; USED history is in memory only: cannot save handeye_config.yml";
}

void newBatch(App& a) {
    if (a.busy || a.sentCount) { a.message = "NEW BATCH unavailable during a cycle"; return; }
    a.processedBlocks.fill(false); resetObservations(a); a.planRefreshReady = false;
    a.message = saveConfig(a) ? "NEW BATCH: all blocks eligible again; waiting for stable camera readings" :
        "New batch reset in memory; settings save failed";
}

void setPickupMode(App& a, bool allowRepeat) {
    if (a.busy || a.sentCount) { a.message = "Change pickup mode after the current cycle finishes"; return; }
    a.allowRepeatPickups = allowRepeat;
    refreshPlans(a);
    bool saved = saveConfig(a);
    a.message = allowRepeat ? "ALLOW REPEATS: handled blocks may be picked again" :
        "UNUSED ONLY: handled blocks are excluded; history retained";
    if (!saved) a.message += " (settings save failed)";
}

string startBlock(const App& a) {
    if (a.calibrationMode) return "Calibration preview mode - robot commands disabled";
    if (a.busy) return "Sending plan - please wait";
    if (a.sentCount) return "Automatic cycle in progress";
    if (!a.serialConnected) return "Connect COM: launch with COM port argument";
    if (a.learnPick || a.learning) return "Finish or cancel block measurement";
    if (a.mapping || a.sampleMode || a.zoneSetup) return "Finish setup before starting";
    if (a.mapped != N || !a.verifiedMap) return "Open SETUP and map / verify P1-P9";
    if (a.commandLatched) return "Previous command still latched - check controller";
    if (!a.planRefreshReady) return "Waiting for fresh camera observations";
    if (!a.queued[a.selected - 1].valid) return a.queued[a.selected - 1].message;
    return "";
}

// Dynamic single-block test: staged operator actions, never automatic retries.
vector<uint8_t> dynamicFields(int token, int action, Point2d xy) {
    if (token < 1 || token>127 || action < 1 || action>4 || !isfinite(xy.x) || !isfinite(xy.y) ||
        xy.x < 392.7 || xy.x>492.6 || xy.y < -62.8 || xy.y>47.2)throw runtime_error("Dynamic XY/action outside test limits");
    int qx = cvRound((xy.x - 350) * 10), qy = cvRound((xy.y + 100) * 10);
    vector<uint8_t> out = { uint8_t(token),uint8_t(action),uint8_t(qx / 128 + 1),uint8_t(qx % 128 + 1),uint8_t(qy / 128 + 1),uint8_t(qy % 128 + 1) };
    int sum = 0;for (auto b : out)sum += b;out.push_back(uint8_t(sum % 251 + 1));return out;
}
bool lockedPickupReady(const App& a) {
    if (!a.calibration || a.dynamicStep != 1 || a.dynamicToken < 1 || a.dynamicToken>127)return false;
    const auto& c = *a.calibration;
    if (!c.valid || c.opt.source != "camera" || c.freeze || c.entering || !c.test || c.planning || c.comparisonView)return false;
    try { dynamicFields(a.dynamicToken, 2, a.dynamicXY); }
    catch (...) { return false; }
    // The operator verifies HOVER COMPLETE and alignment. The arm may hide the block.
    return true;
}
constexpr double HOVER_SETTLE_S = 25.0;   // robot needs this long to reach hover; a PICK frame sent earlier is lost (no ACK)
void dynamicAction(App& a, int action) {
    if (a.busy || !a.calibrationMode || !a.dynamicTestMode || !a.calibration)return;
    auto& c = *a.calibration;
    if (!a.serialConnected || a.sentCount) { a.dynamicStatus = "Blocked: COM offline or stack cycle active";return; }
    if (action == 1 || action == 2 || action == 4) {
        if (!c.valid || c.opt.source != "camera" || c.freeze || c.entering || !c.test || c.planning || c.comparisonView) { a.dynamicStatus = "Need LIVE Coordinate Test with a valid real-camera mapping";return; }
    }
    if (action == 1 || action == 4) {
        if (a.dynamicStep != 0 || !c.selectStable || c.detections.empty()) { a.dynamicStatus = "Select one stationary block first";return; }
        auto d = c.detections[c.selected];auto xy = calibrationui::project(c.mapping.h, d.centre) + a.dynamicTrim;
        if (calibrationui::regionDistance(c.mapping, xy) < 20) { a.dynamicStatus = "Move test block at least 20mm inside centre boundary";return; }
        for (size_t j = 0;j < c.detections.size();++j)if (int(j) != c.selected &&
            norm(calibrationui::project(c.mapping.h, c.detections[j].centre) - xy) < 60) {
            a.dynamicStatus = "Isolate selected block: other detected blocks need 60mm clearance";return;
        }
        // Initial test permits faces aligned with robot X/Y only (10-degree tolerance).
        for (int i = 0;i < 4;++i) {
            Point2d edge = calibrationui::project(c.mapping.h, d.quad[(i + 1) % 4]) - calibrationui::project(c.mapping.h, d.quad[i]);
            double angle = atan2(abs(edge.y), abs(edge.x)) * 180 / CV_PI;if (min(angle, 90 - angle) > 10) { a.dynamicStatus = "Align block faces with original grid axes for this test";return; }
        }
        a.dynamicXY = Point2d(cvRound(xy.x * 10) / 10.0, cvRound(xy.y * 10) / 10.0);
        a.dynamicToken = int(chrono::steady_clock::now().time_since_epoch().count() % 127) + 1;a.dynamicColour = d.colour;
    }
    else if (action == 2) {
        if (!lockedPickupReady(a)) { a.dynamicStatus = "Need a valid locked HOVER session; verify pendant and alignment before PICK";return; }
        double since = chrono::duration<double>(chrono::steady_clock::now() - a.hoverSentAt).count();
        if (since < HOVER_SETTLE_S) { a.dynamicStatus = "WAIT " + to_string(int(ceil(HOVER_SETTLE_S - since))) + " s: robot is still moving to hover; a PICK sent now would be lost. Wait for pendant HOVER COMPLETE";return; }
        // Use immutable pre-hover XY/token; NEVER retarget from arm-occluded live detections.
    }
    else if (action == 3) {
        if (a.dynamicStep != 1 && a.dynamicStep != 2) { a.dynamicStatus = "No active hover/pick session";return; }
    }
    else return;
    try {
        auto fields = dynamicFields(a.dynamicToken, action, a.dynamicXY);
        struct Guard { bool& b;Guard(bool& v) :b(v) { b = true; }~Guard() { b = false; } } guard(a.dynamicSending);
        bool ok = sendFrame(a, 230, fields, 231);
        if (!ok) { a.dynamicStatus = "TX FAILED: robot state unknown. Use controller stop/recovery; no automatic retry.";return; }
        if (action == 1) a.hoverSentAt = chrono::steady_clock::now();
        a.dynamicStep = action == 4 ? 2 : action;
        a.dynamicStatus = action == 4 ? "GO + PICK SENT: XY was locked BEFORE motion. Watch the pendant for PICK/LIFT COMPLETE, then click REPLACE + HOME" : action == 1 ? "HOVER SENT. Wait for pendant HOVER COMPLETE, check the gripper is over the block by eye, then click 2 CONFIRM ALIGNED: PICK" :
            action == 2 ? "PICK SENT: verify pendant PICK/LIFT COMPLETE before REPLACE" :
            "HOME/REPLACE SENT: verify robot is home and block released, then click VERIFIED HOME";
        cout << a.dynamicStatus << " X=" << a.dynamicXY.x << " Y=" << a.dynamicXY.y << endl;
    }
    catch (const exception& e) { a.dynamicStatus = e.what(); }
}

bool switchPage(App& a, int page) {
    if (a.busy || a.sentCount || a.dynamicStep) { a.dynamicStatus = "Finish current robot session before changing pages";a.message = a.dynamicStatus;return false; }
    if (!a.calibration)return false;
    auto& c = *a.calibration;
    if (c.entering) { c.message = "Finish or cancel coordinate entry before changing pages";return false; }
    a.calibrationMode = page != 0;a.dynamicTestMode = page == 2;c.exit = false;c.freeze = false;c.comparisonView = false;c.planning = false;
    a.mapping = false;a.sampleMode = false;a.learnPick = false;a.learning = false;a.zoneSetup = 0;
    if (page == 1) { c.test = false;c.message = "CALIBRATE: verify references/plane, collect stable points, Fit + save"; }
    if (page == 2) {
        c.test = true;c.testStable = 0;c.selectStable = false;
        a.dynamicStatus = c.valid ? "TEST: select isolated aligned block. Send HOVER first; check pendant before PICK." :
            "TEST LOCKED: open CALIBRATE, Fit + Save first. Saved mappings are not automatically loaded after restart.";
        c.message = a.dynamicStatus;
    }
    if (page == 0) { resetObservations(a);a.planRefreshReady = false;a.message = "OPERATE: fixed-coordinate demonstration; camera refreshing"; }
    return true;
}
void drawNavigation(Mat& ui, const App& a) {
    fill(ui, Rect(0, 0, 1280, 90), Scalar(30, 26, 23));
    label(ui, "HAND-EYE  /  OPERATOR CONSOLE", Point(20, 28), .65, WHITE, 2);
    const char* names[] = { "OPERATE","CALIBRATE","TEST" };int current = a.calibrationMode ? (a.dynamicTestMode ? 2 : 1) : 0;
    for (int i = 0;i < 3;++i)button(ui, Rect(20 + i * 195, 45, 185, 34), names[i], current == i);
    if (current == 1) {
        button(ui, Rect(660, 45, 180, 34), "FIT + SAVE");button(ui, Rect(850, 45, 180, 34), "INVALIDATE");button(ui, Rect(1040, 45, 220, 34), "SNAPSHOT");
        label(ui, "Preview only / no robot commands", Point(750, 27), .40, GOLD);
    }
    else if (current == 2) {
        button(ui, Rect(660, 45, 180, 34), "9 POINT CHECK");button(ui, Rect(850, 45, 180, 34), "RECORD ACTUAL XY");button(ui, Rect(1040, 45, 220, 34), "SNAPSHOT");
        label(ui, a.calibration && a.calibration->valid ? "SUPERVISED TEST / buttons can command motion" : "TEST LOCKED / Fit + Save calibration first", Point(690, 27), .40, GOLD);
    }
    else label(ui, "Fixed-coordinate demo / physical STOP on controller", Point(660, 65), .40, MUTED);
}

// ---------------------------------------------------------------- mouse
void mouse(int event, int x, int y, int, void* ptr) {
    if (event != EVENT_LBUTTONDOWN) return;
    App& a = *static_cast<App*>(ptr);
    if (a.busy)return;
    if (y >= 45 && y < 79) {
        for (int i = 0;i < 3;++i)if (Rect(20 + i * 195, 45, 185, 34).contains(Point(x, y))) { switchPage(a, i);return; }
        if (a.calibrationMode) {
            if (a.dynamicStep)return;
            const char calActions[] = { 'c','i','s' }, testActions[] = { 'x','e','s' };
            for (int i = 0;i < 3;++i)if (Rect(660 + i * 190, 45, i == 2 ? 220 : 180, 34).contains(Point(x, y))) { try { a.calibration->action(a.dynamicTestMode ? testActions[i] : calActions[i]); } catch (const exception& e) { a.calibration->message = e.what(); }return; }
            return;
        }
    }
    if (a.calibrationMode && a.dynamicTestMode && y >= 775 && y < 805) {
        if (a.dynamicStep)return;
        const double changes[] = { -.5,.5,-.5,.5 };
        for (int i = 0;i < 4;++i)if (Rect(440 + i * 95, 775, 85, 28).contains(Point(x, y))) {
            double& value = i < 2 ? a.dynamicTrim.x : a.dynamicTrim.y;
            value = max(-5.0, min(5.0, value + changes[i]));saveConfig(a);
            a.dynamicStatus = "XY trim changed: verify with HOVER before pickup";return;
        }
        if (Rect(830, 775, 150, 28).contains(Point(x, y))) { a.dynamicTrim = {};saveConfig(a);return; }
    }
    if (a.calibrationMode && y >= 845 && y < 875) {
        if (a.dynamicTestMode) {
            if (Rect(20, 845, 200, 30).contains(Point(x, y))) { a.dynamicRequest = 4;return; }
            if (Rect(230, 845, 220, 30).contains(Point(x, y))) { a.dynamicRequest = 1;return; }
            if (Rect(460, 845, 220, 30).contains(Point(x, y))) { a.dynamicRequest = 2;return; }
            if (Rect(690, 845, 220, 30).contains(Point(x, y))) { a.dynamicRequest = 3;return; }
            if (Rect(920, 845, 340, 30).contains(Point(x, y))) {
                if (a.dynamicStep == 3) { a.dynamicStep = 0;a.dynamicStatus = "Operator confirmed HOME/release. Single-block test finished"; }
                return;
            }
        }
    }
    if (a.calibrationMode && a.dynamicStep) { a.dynamicStatus = "Session locked: use PICK, CANCEL/REPLACE or VERIFIED HOME; controller STOP for faults";return; }
    if (a.calibrationMode && a.calibration) {
        try { a.calibration->click(x, y); }
        catch (const exception& e) { a.calibration->message = e.what(); }
        return;
    }
    if (a.busy) return;
    if (Rect(895, 574, 175, 30).contains(Point(x, y))) { setPickupMode(a, false); return; }
    if (Rect(1078, 574, 175, 30).contains(Point(x, y))) { setPickupMode(a, true); return; }
    if (Rect(895, 640, 355, 58).contains(Point(x, y))) { string reason = startBlock(a); if (reason.empty()) operatorStart(a); else a.message = reason; return; }
    if (Rect(1080, 92, 170, 34).contains(Point(x, y)) && a.sentCount == 0) {
        a.setupOpen = !a.setupOpen; a.mapping = false; a.sampleMode = false; a.zoneSetup = 0; a.learnPick = false; a.learning = false; return;
    }
    if (a.setupOpen && a.sentCount == 0) {
        const string keys = "mvrgbpzxk";
        if (Rect(25, 833, 190, 28).contains(Point(x, y))) { resetSettings(a, false); return; }
        if (Rect(225, 833, 200, 28).contains(Point(x, y))) { resetSettings(a, true); return; }
        if (Rect(1060, 833, 190, 28).contains(Point(x, y))) { newBatch(a); return; }
        if (Rect(435, 833, 220, 28).contains(Point(x, y))) {
            a.pendingKey = 'l'; return;
        }
        for (int i = 0; i < 9; ++i)
            if (Rect(25 + i * 138, 795, 129, 38).contains(Point(x, y))) { a.pendingKey = keys[i]; return; }
    }
    if (a.sentCount > 0) return;
    if (a.learnPick && a.imageRect.contains(Point(x, y))) {
        Point pixel((x - a.imageRect.x) * a.cameraSize.width / a.imageRect.width,
            (y - a.imageRect.y) * a.cameraSize.height / a.imageRect.height);
        for (const auto& d : a.lastDetections) if (d.box.contains(pixel) && d.measurable) {
            a.learnCentre = d.centre; a.learnColour = d.colour;
            a.learnPick = false; a.learning = true; a.sizeSamples.clear();
            a.message = "Measuring selected 40x40mm face; keep still for 30 frames"; return;
        }
        a.message = "Click an isolated, fully visible square block face"; return;
    }
    if (a.zoneSetup > 0 && a.imageRect.contains(Point(x, y))) {
        int xx = (x - a.imageRect.x) * a.cameraSize.width / a.imageRect.width;
        int yy = (y - a.imageRect.y) * a.cameraSize.height / a.imageRect.height;
        int j = a.zoneSetup - 1;
        a.stackZones[j] = Point(xx, yy); a.zoneDefined[j] = true; a.zoneSetup = 0;
        a.stackTop[j] = 0; a.observedHistory[j].clear(); saveConfig(a);
        a.message = "Saved S" + to_string(j + 1) + " stack zone; monitor visible top only";
        return;
    }
    if (a.mapping && a.imageRect.contains(Point(x, y)) && a.mapped < N) {
        int xx = (x - a.imageRect.x) * a.cameraSize.width / a.imageRect.width;
        int yy = (y - a.imageRect.y) * a.cameraSize.height / a.imageRect.height;
        a.points[a.mapped++] = Point(xx, yy);
        a.message = "Mapped P" + to_string(a.mapped);
        if (a.mapped == N) { a.mapping = false; a.verifiedMap = true; saveConfig(a); a.message = "Manual P1-P9 reference verified and saved"; }
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
        if (h.size() < 10) { a.message = "Sample failed: click the coloured block centre"; return; }
        double sx = 0, sy = 0;
        for (int hue : h) { double angle = hue * 2 * CV_PI / 180.; sx += cos(angle); sy += sin(angle); }
        double ang = atan2(sy, sx); if (ang < 0) ang += 2 * CV_PI;
        a.hue[a.sampleColour] = int(round(ang * 180 / (2 * CV_PI))) % 180;
        a.tolerance[a.sampleColour] = 12; a.sampleMode = false; resetObservations(a); saveConfig(a);
        a.message = string("Sampled ") + cname(a.sampleColour) + " hue=" + to_string(a.hue[a.sampleColour]);
        return;
    }
    if (y >= 158 && y <= 196 && x >= 887 && x <= 1255) {
        int n = (x - 887) / 124;
        if (n >= 0 && n < 3) { a.selected = n + 1; a.releaseArmed = false; a.message = "Selected S" + to_string(a.selected); }
    }
}

void drawStack(Mat& ui, int x, int y, int routine) {
    const array<array<int, 3>, 3> top{ {{{GREEN,RED,BLUE}},{{RED,GREEN,BLUE}},{{BLUE,GREEN,RED}}} };
    for (int i = 0; i < 3; ++i) {
        Rect r(x, y + i * 34, 115, 31);
        fill(ui, r, ink(top[routine - 1][i]));
        label(ui, cname(top[routine - 1][i]), Point(x + 13, y + i * 34 + 21), .43, Scalar(255, 255, 255), 2);
    }
}

int measurementSelfTest() {
    const auto previous = fileutil::currentPath();
    try {
        string sandbox = previous + "/measurement_test_output";
        fileutil::createDirectory(sandbox); fileutil::currentPath(sandbox);
        for (int action = 1;action <= 3;++action)for (auto xy : vector<Point2d>{ {392.7,-62.8},{442.65,-7.8},{492.6,47.2} }) {
            auto b = dynamicFields(17, action, xy);int sum = 0;for (int i = 0;i < 6;++i) { if (!b[i])throw runtime_error("Zero dynamic field");sum += b[i]; }
            Point2d decoded(350 + ((b[2] - 1) * 128 + b[3] - 1) / 10.0, -100 + ((b[4] - 1) * 128 + b[5] - 1) / 10.0);
            if (b.size() != 7 || b[6] != sum % 251 + 1 || norm(decoded - xy) > .071)throw runtime_error("Dynamic protocol roundtrip failed");
        }
        bool rejected = false;try { dynamicFields(1, 1, { 600,0 }); }
        catch (...) { rejected = true; }if (!rejected)throw runtime_error("Out of range dynamic accepted");
        App occluded;calibrationui::App occludedCamera;occluded.calibration = &occludedCamera;
        occludedCamera.valid = true;occludedCamera.opt.source = "camera";occludedCamera.test = true;occludedCamera.selectStable = false;
        occluded.dynamicStep = 1;occluded.dynamicToken = 42;occluded.dynamicXY = { 432.1,24.9 };
        auto locked = occluded.dynamicXY;
        if (!lockedPickupReady(occluded) || occluded.dynamicXY != locked)throw runtime_error("Occlusion blocked or altered locked pickup");
        occluded.dynamicStep = 0;if (lockedPickupReady(occluded))throw runtime_error("Pickup allowed without hover");
        occluded.dynamicStep = 1;occludedCamera.valid = false;if (lockedPickupReady(occluded))throw runtime_error("Pickup allowed with invalid mapping");
        cout << "PASS: occluded pickup uses locked XY; missing hover/invalid mapping still blocked\n";
        App stageGuard;calibrationui::App testCalibration;stageGuard.calibration = &testCalibration;stageGuard.calibrationMode = true;stageGuard.dynamicTestMode = true;
        dynamicAction(stageGuard, 2);if (stageGuard.dynamicStep || stageGuard.dynamicSending)throw runtime_error("Invalid dynamic stage changed state");
        cout << "PASS: dynamic base128 XY encoding/checksum, bounds rejection, guarded single-block test entry\n";
        App navigation;calibrationui::App navCalibration;navigation.calibration = &navCalibration;
        if (!switchPage(navigation, 2) || !navigation.calibrationMode || navigation.calibration->valid || navigation.dynamicStatus.find("TEST LOCKED") == string::npos)throw runtime_error("Uncalibrated Test access/lock failed");
        if (!switchPage(navigation, 1) || !navigation.calibrationMode || navigation.dynamicTestMode)throw runtime_error("Calibration navigation failed");
        navCalibration.valid = true;
        if (!switchPage(navigation, 2) || !navigation.dynamicTestMode)throw runtime_error("Test navigation failed");
        navigation.dynamicStep = 1;if (switchPage(navigation, 0))throw runtime_error("Active dynamic session allowed page change");
        navigation.dynamicStep = 0;if (!switchPage(navigation, 0) || navigation.calibrationMode)throw runtime_error("Operate navigation failed");
        cout << "PASS: Operate/Calibrate/Test navigation and active-session locks\n";
        App guard; guard.calibrationMode = true; guard.serialConnected = true; guard.lastSent = 91;
        if (sendCommand(guard, 240) || guard.lastSent != 91) throw runtime_error("serial calibration guard failed");
        operatorStart(guard); operatorReturn(guard); operatorClear(guard); operatorRelease(guard);
        if (guard.sentCount || guard.lastSent != 91) throw runtime_error("operator calibration guards failed");
        if (startBlock(guard).find("Calibration") == string::npos) throw runtime_error("calibration start not blocked");
        cout << "PASS: integrated calibration blocks serial writes and all operator motion commands\n";
        App a; a.mapped = N; a.verifiedMap = true; a.zoneDefined.fill(true);
        a.hue[GREEN] = 57; a.learnedFace = Size2f(60, 62); a.learnedResolution = Size(850, 620);
        resetSettings(a, false);
        App loaded; loadConfig(loaded);
        if (loaded.mapped || loaded.verifiedMap || loaded.autoProposal || loaded.zoneDefined[0] || loaded.hue[GREEN] != 57 || loaded.learnedFace.width != 60)
            throw runtime_error("mapping reset persistence failed");
        a.sentCount = 1; a.mapped = 9; resetSettings(a, true);
        if (a.mapped != 9) throw runtime_error("active cycle reset guard failed");
        a.sentCount = 0; resetSettings(a, true); App defaults; loadConfig(defaults);
        if (defaults.hue[GREEN] != 60 || defaults.learnedFace.width != 0 || defaults.autoProposal)
            throw runtime_error("full reset persistence failed");
        Mat frame(400, 500, CV_8UC3, Scalar(20, 20, 20));
        RotatedRect expected(Point2f(200, 180), Size2f(64, 64), 23);
        Point2f vertices[4]; expected.points(vertices); vector<Point> polygon;
        for (auto v : vertices) polygon.push_back(Point(cvRound(v.x), cvRound(v.y)));
        fillConvexPoly(frame, polygon, Scalar(0, 0, 255));
        Mat hsv; cvtColor(frame, hsv, COLOR_BGR2HSV); Mat overlay = frame.clone(); vector<Detection> det;
        findColour(hsv, overlay, RED, a, det);
        if (det.size() != 1 || !det[0].measurable || abs(det[0].face.size.width - 64) > 3 || abs(det[0].face.size.height - 64) > 3)
            throw runtime_error("rotated block measurement failed");
        a.cameraSize = frame.size(); a.lastFrame = frame; a.learnCentre = det[0].centre; a.learnColour = RED; a.learning = true;
        for (int i = 0;i < SIZE_SAMPLES;++i) updateBlockCapture(a, det);
        if (a.learning || a.learnedFace.width < 60 || !fileutil::exists(a.sizeCapturePath + "/samples.csv") || !fileutil::exists(a.sizeCapturePath + "/camera.png"))
            throw runtime_error("capture/export failed");
        ifstream samples(a.sizeCapturePath + "/samples.csv"); string row; int rows = 0;
        while (getline(samples, row)) ++rows;
        if (rows != 31) throw runtime_error("wrong sample count");
        App restored; loadConfig(restored);
        if (restored.learnedResolution != frame.size() || restored.learnedFace.width != a.learnedFace.width)
            throw runtime_error("measurement reload failed");
        a.learning = true; a.learnCentre = det[0].centre; a.sizeSamples.clear();
        updateBlockCapture(a, det); updateBlockCapture(a, {});
        if (!a.sizeSamples.empty()) throw runtime_error("lost face did not restart stable capture");
        App batch; batch.autoProposal = false;
        for (int i = 0;i < N;++i) { batch.positions[i].uncertain = false; batch.positions[i].confirmed = i % 3 + 1; }
        Plan initial = makePlan(batch, 1);
        if (!initial.valid) throw runtime_error("initial unused plan missing");
        batch.selected = 1; batch.sentPlans[0] = true; batch.returnSent[0] = true;
        batch.sentSources[0] = initial.source; batch.sentCount = 1; batch.sendOrder[0] = 1;
        // Return the three colours to a different permutation of the vacated slots.
        batch.returnDest[0] = { {initial.source[1],initial.source[2],initial.source[0]} };
        auto destinations = batch.returnDest[0];
        operatorRelease(batch);
        for (int i = 0;i < N;++i) { batch.positions[i].uncertain = false; batch.positions[i].confirmed = i % 3 + 1; }
        for (int l = 0;l < 3;++l) batch.positions[destinations[l]].confirmed = initial.colours[l];
        Plan next = makePlan(batch, 1);
        if (!next.valid) throw runtime_error("unused blocks not available after cycle");
        for (int src : next.source) if (batch.processedBlocks[src]) throw runtime_error("planner reselected USED block");
        App persisted; loadConfig(persisted);
        for (int dest : destinations) if (!persisted.processedBlocks[dest]) throw runtime_error("USED history persistence failed");
        batch.processedBlocks.fill(true);
        if (makePlan(batch, 1).valid) throw runtime_error("all USED blocks still selectable");
        auto historyBeforeMode = batch.processedBlocks;
        setPickupMode(batch, true);
        if (!makePlan(batch, 1).valid || batch.processedBlocks != historyBeforeMode)
            throw runtime_error("repeat mode failed or destroyed history");
        App modeLoaded; loadConfig(modeLoaded);
        if (!modeLoaded.allowRepeatPickups) throw runtime_error("repeat mode did not persist");
        batch.sentCount = 1; setPickupMode(batch, false);
        if (!batch.allowRepeatPickups) throw runtime_error("mode changed during active cycle");
        batch.sentCount = 0; setPickupMode(batch, false);
        if (makePlan(batch, 1).valid || batch.processedBlocks != historyBeforeMode)
            throw runtime_error("unused mode did not restore exclusion");
        cout << "PASS: repeat/unused modes, saved preference, history preserved, active-cycle mode guard\n";
        batch.sentCount = 1; newBatch(batch);
        if (!batch.processedBlocks[0]) throw runtime_error("new batch allowed during active cycle");
        batch.sentCount = 0; newBatch(batch);
        for (bool used : batch.processedBlocks) if (used) throw runtime_error("new batch did not clear USED history");
        cout << "PASS: no-repeat planner, random-return history transfer, persistence, exhausted batch, guarded NEW BATCH reset\n";
        cout << "PASS: rotated contour size, 30-frame capture, CSV/image/YAML export and reload, persistent map/full resets, cycle reset guard, lost-face handling\n";
        fileutil::currentPath(previous); return 0;
    }
    catch (const exception& e) { fileutil::currentPath(previous); cerr << e.what() << endl; return 1; }
}

// ---------------------------------------------------------------- main
static const string BUILD_TAG = string("BUILD GOPICK-V2 ") + __DATE__ + " " + __TIME__;
int main(int argc, char** argv) {
    if (argc >= 2 && string(argv[1]) == "--calibration-self-test") { try { return calibrationui::selfTest() ? 0 : 1; } catch (const exception& e) { cerr << e.what() << endl; return 1; } }
    if (argc >= 2 && string(argv[1]) == "--self-test") return measurementSelfTest();
    cout << BUILD_TAG << endl;
    App a; loadConfig(a);
    calibrationui::App calibration; calibration.opt.source = "camera"; calibration.opt.camera = CAM;
    calibration.buttons.back().label = "Q Back";
    a.calibration = &calibration;
    if (fileutil::exists("calibration_reference_points.csv") &&
        !calibrationui::loadReferences("calibration_reference_points.csv", calibration.refs)) {
        cerr << "Invalid calibration_reference_points.csv" << endl; return 1;
    }
    bool previewCalibration = argc >= 2 && string(argv[1]) == "--preview-calibration";
    if (!previewCalibration && calibration.loadPersistentCalibration()) cout << "Loaded saved calibration (calibration_saved.yml): TEST is ready, no recalibration needed.\n";
    bool preview = previewCalibration || (argc >= 2 && (string(argv[1]) == "--preview" || string(argv[1]) == "--preview-setup"));
    if (preview) { a = App{}; a.setupOpen = string(argv[1]) == "--preview-setup"; a.autoProposal = false; }
    a.calibration = &calibration;
    if (previewCalibration) { a.calibrationMode = true; calibration.opt.source = "demo"; }
    if (argc >= 2 && !preview) startSerial(a, argv[1]);
    else cout << "Usage: HandEye_COM_Direct.exe COM3\n";

    VideoCapture cap;
    if (!preview) cap.open(CAM);
    if (!preview && !cap.isOpened()) { cerr << "Cannot open camera index " << CAM << "\n"; disconnect(a); return 1; }
    namedWindow("HAND EYE | HD Dashboard", WINDOW_NORMAL);
    resizeWindow("HAND EYE | HD Dashboard", W, H);
    setMouseCallback("HAND EYE | HD Dashboard", mouse, &a);
    cout << "CYCLE: 1/2/3 select -> S send stack -> automatic home / random return / release.\n"
        << "Map once with M (or auto-propose with nine blocks); V verifies.\n"
        << "R/G/B sample colour then click block, K reset HSV, P refresh plans, Q quit.\n"
        << "Z then 1/2/3 then click stack zone; X clears stack observation history.\n";

    while (true) {
        Mat frame;
        if (preview) {
            frame = Mat(620, 850, CV_8UC3, Scalar(34, 39, 45));
            for (int i = 0; i < 9; ++i) {
                Point centre(180 + (i % 3) * 180, 150 + (i / 3) * 160);
                rectangle(frame, Rect(centre.x - 30, centre.y - 30, 60, 60), ink(i % 3 + 1), FILLED);
                a.points[i] = centre;
            }
            a.mapped = N; a.verifiedMap = true;
        }
        else cap >> frame;
        if (frame.empty()) break;
        a.cameraSize = frame.size();
        if (a.calibrationMode) {
            if (previewCalibration) {
                for (int i = 0;i < 25;++i) { calibration.frame = calibrationui::synthetic(calibration); calibration.update(); }
            }
            if (!calibration.freeze && !previewCalibration) { calibration.frame = frame.clone(); calibration.update(); }
            if (a.dynamicRequest) { int request = a.dynamicRequest;a.dynamicRequest = 0;dynamicAction(a, request); }
            calibration.draw();
            drawNavigation(calibration.ui, a);
            if (a.dynamicTestMode) {
                fill(calibration.ui, Rect(0, 752, 1280, 126), Scalar(30, 26, 23));
                label(calibration.ui, "TEST trim X " + calibrationui::fmt(a.dynamicTrim.x, 1) + " Y " + calibrationui::fmt(a.dynamicTrim.y, 1) + " mm", Point(20, 797), .42, GOLD);
                const char* trimNames[] = { "X -0.5","X +0.5","Y -0.5","Y +0.5" };
                for (int i = 0;i < 4;++i)button(calibration.ui, Rect(440 + i * 95, 775, 85, 28), trimNames[i]);
                button(calibration.ui, Rect(830, 775, 150, 28), "RESET TRIM");
                button(calibration.ui, Rect(20, 845, 200, 30), "0 GO + PICK (1 CLICK)", a.dynamicStep == 0 && calibration.valid);
                button(calibration.ui, Rect(230, 845, 220, 30), "1 SEND HOVER", a.dynamicStep == 0 && calibration.valid);
                string pickLabel = "2 CONFIRM ALIGNED: PICK";bool pickReady = a.dynamicStep == 1;
                if (pickReady) { double since = chrono::duration<double>(chrono::steady_clock::now() - a.hoverSentAt).count();if (since < HOVER_SETTLE_S) { pickLabel = "2 PICK (wait " + to_string(int(ceil(HOVER_SETTLE_S - since))) + " s)";pickReady = false; } }
                button(calibration.ui, Rect(460, 845, 220, 30), pickLabel, pickReady);
                button(calibration.ui, Rect(690, 845, 220, 30), a.dynamicStep == 1 ? "3 CANCEL + HOME" : "3 REPLACE + HOME", a.dynamicStep == 1 || a.dynamicStep == 2);
                button(calibration.ui, Rect(920, 845, 340, 30), "4 I VERIFIED HOME + RELEASE", a.dynamicStep == 3);
                if (calibration.valid && !calibration.detections.empty()) {
                    auto target = a.dynamicStep ? a.dynamicXY : calibrationui::project(calibration.mapping.h, calibration.detections[calibration.selected].centre) + a.dynamicTrim;
                    fill(calibration.ui, Rect(855, 653, 410, 28), Scalar(48, 43, 38));
                    label(calibration.ui, "COMMAND X " + calibrationui::fmt(target.x, 1) + " Y " + calibrationui::fmt(target.y, 1) + " mm", Point(865, 674), .42, GOLD);
                }
                fill(calibration.ui, Rect(0, 810, 1280, 30), Scalar(30, 26, 23));
                label(calibration.ui, a.dynamicStatus, Point(18, 832), .40, GOLD);
            }
            label(calibration.ui, a.dynamicTestMode ? "TEST: fixed Z200.81 / approach286.03 mm. No ACK. Check pendant before each next action." : "O: BACK TO OPERATION | Calibration/test only - no robot commands", Point(20, a.dynamicTestMode ? 893 : 880), .43, GOLD);
            if (previewCalibration) { imwrite("operator_calibration_preview.png", calibration.ui); break; }
            label(calibration.ui, BUILD_TAG, Point(900, 893), .38, MUTED);
            imshow("HAND EYE | HD Dashboard", calibration.ui);
            int ck = waitKey(25);
            if (a.dynamicStep) {
                if (ck >= 0)a.dynamicStatus = "Session active: use staged test buttons; physical controller STOP for faults";
                continue;
            }
            if (!calibration.entering && ck >= '1' && ck <= '3') { switchPage(a, ck - '1');continue; }
            if (calibration.exit || (!calibration.entering && (ck == 'o' || ck == 'O' || ck == 'q' || ck == 'Q' || ck == 27))) {
                switchPage(a, 0);
            }
            else { try { calibration.key(ck); } catch (const exception& e) { calibration.message = e.what(); } }
            continue;
        }
        Mat smoothed, hsv;
        GaussianBlur(frame, smoothed, Size(5, 5), 0);
        cvtColor(smoothed, hsv, COLOR_BGR2HSV);
        a.lastHSV = hsv;
        Mat camera = frame.clone();
        vector<Detection> det;
        for (int c = 1; c <= 3; ++c) findColour(hsv, camera, c, a, det);
        a.lastDetections = det; a.lastFrame = frame;
        updateBlockCapture(a, det);

        if (a.autoProposal && a.mapped != N && !a.mapping && !a.planLocked && !a.learnPick && !a.learning) {
            array<Point, N> proposed{};
            if (proposeGrid(det, proposed)) {
                bool consistent = a.autoCandidate;
                if (consistent) for (int i = 0; i < N; ++i) if (norm(proposed[i] - a.candidatePoints[i]) > 15) consistent = false;
                a.candidateStreak = consistent ? a.candidateStreak + 1 : 1;
                a.candidatePoints = proposed; a.autoCandidate = true;
                if (a.candidateStreak >= STABLE) {
                    a.points = proposed; a.mapped = N; a.verifiedMap = false; resetObservations(a);
                    a.message = "AUTO GRID PROPOSAL: verify P1-P9 labels, press V to accept";
                }
            }
            else { a.autoCandidate = false; a.candidateStreak = 0; }
        }
        if (a.mapped == N && !a.mapping) {
            auto matched = matchDetections(a, det);
            for (int i = 0; i < N; ++i) {
                int observation = matched[i] >= 0 ? det[matched[i]].colour : UNKNOWN;
                auto& s = a.positions[i];
                if (observation == UNKNOWN) {
                    s.candidate = UNKNOWN; s.streak = 0;
                    if (s.missing < LOST_FRAMES) s.missing++;
                    s.uncertain = s.missing < LOST_FRAMES;
                    if (s.missing >= LOST_FRAMES) s.confirmed = UNKNOWN;
                }
                else {
                    s.missing = 0;
                    if (s.candidate == observation) s.streak++;
                    else { s.candidate = observation; s.streak = 1; }
                    s.uncertain = (s.streak < STABLE);
                    if (s.streak >= STABLE) s.confirmed = observation;
                }
            }
        }
        if (a.mapped == N && a.verifiedMap &&
            (!a.planRefreshReady || chrono::duration_cast<chrono::seconds>(
                chrono::steady_clock::now() - a.lastPlanRefresh).count() >= 5)) {
            refreshPlans(a);
        }
        for (int j = 0; j < 3; ++j) if (a.zoneDefined[j]) {
            int seen = UNKNOWN; double best = 0;
            for (const auto& d : det) {
                double distance = norm(d.centre - a.stackZones[j]);
                if (distance < 55 && d.area > best) { best = d.area; seen = d.colour; }
            }
            if (a.topCandidate[j] == seen) a.topStreak[j]++; else { a.topCandidate[j] = seen; a.topStreak[j] = 1; }
            if (a.topStreak[j] >= STABLE) {
                if (seen != a.stackTop[j]) {
                    if (seen != UNKNOWN && (a.observedHistory[j].empty() || a.observedHistory[j].back() != seen))
                        a.observedHistory[j].push_back(seen);
                    a.stackTop[j] = seen;
                }
            }
            drawMarker(camera, a.stackZones[j], GOLD, MARKER_CROSS, 12, 1);
            label(camera, "STACK S" + to_string(j + 1) + " TOP:" + cname(a.stackTop[j]), a.stackZones[j] + Point(-52, -62), .43, Scalar(230, 205, 75), 2);
        }
        // Fresh live observations, independent of the 5-second planning refresh.
        if (a.autoMode && !a.busy) {
            const auto now = chrono::steady_clock::now();
            const chrono::steady_clock::time_point unset{};
            for (int j = 0; j < 3; ++j) {
                if (!a.sentPlans[j]) continue;
                const int saved = a.selected;
                a.selected = j + 1;
                if (!a.returnSent[j]) {
                    int empty = 0;
                    bool sourcesEmpty = true;
                    for (const auto& p : a.positions)
                        if (!p.uncertain && p.confirmed == UNKNOWN) ++empty;
                    for (int src : a.sentSources[j])
                        if (src < 0 || a.positions[src].uncertain || a.positions[src].confirmed != UNKNOWN)
                            sourcesEmpty = false;
                    if (sourcesEmpty && empty >= 3) {
                        if (a.emptySince[j] == unset) a.emptySince[j] = now;
                        if (chrono::duration<double>(now - a.emptySince[j]).count() >= 3.0)
                            operatorReturn(a);
                    }
                    else a.emptySince[j] = unset;
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
                        if (chrono::duration<double>(now - a.restoredSince[j]).count() >= 5.0) {
                            operatorClear(a);
                            operatorRelease(a);
                        }
                    }
                    else a.restoredSince[j] = unset;
                    // No home ACK exists. Retry IDENTICAL destinations until return begins.
                    if (a.sentPlans[j] && !a.returnStarted[j] &&
                        chrono::duration<double>(now - a.retryAt[j]).count() >= 15.0) {
                        const auto& d = a.returnDest[j];
                        const vector<uint8_t> fields = {
                            static_cast<uint8_t>(0x80 + j + 1),
                            static_cast<uint8_t>(0x10 + d[2] + 1),
                            static_cast<uint8_t>(0x20 + d[1] + 1),
                            static_cast<uint8_t>(0x30 + d[0] + 1) };
                        sendFrame(a, 250, fields, 251);
                        a.retryAt[j] = chrono::steady_clock::now();
                    }
                }
                a.selected = saved;
            }
        }
        for (int i = 0; i < a.mapped; ++i) {
            Point p = a.points[i]; const Position& st = a.positions[i];
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
        label(ui, "PLANNED / SENT ONLY - NO LIVE ABB FEEDBACK", Point(890, 43), .43, GOLD, 2);
        panel(ui, Rect(10, 86, 860, 648));
        Mat camFit; resize(camera, camFit, Size(CAMW, CAMH)); camFit.copyTo(ui(a.imageRect));
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
        for (int j = 0; j < 3; ++j) if (a.sentPlans[j]) active = j;
        string phase = "READY TO START";
        if (active >= 0) phase = !a.returnSent[active] ? "PLAN SENT / WATCHING PICKUPS" :
            a.returnStarted[active] ? "RETURNING BLOCKS" : "RETURN PLAN SENT / WAITING";
        else if (!startBlock(a).empty()) phase = "WAITING FOR SETUP / VISION";
        if (a.busy) phase = "SENDING PLAN";
        fill(ui, Rect(892, 210, 362, 55), Scalar(45, 66, 75));
        label(ui, phase, Point(901, 243), .46, GOLD, 2);
        label(ui, "STACK S" + to_string(active >= 0 ? active + 1 : routine) + "  /  TOP TO BOTTOM", Point(895, 291), .47, WHITE, 2);
        drawStack(ui, 895, 305, active >= 0 ? active + 1 : routine);
        label(ui, "PICKUP: BOTTOM FIRST", Point(1024, 305), .37, MUTED);
        for (int l = 0; l < 3; ++l) {
            int src = active >= 0 ? a.sentSources[active][l] : p.source[l];
            label(ui, "L" + to_string(l + 1) + "  " + (src >= 0 ? "P" + to_string(src + 1) : "waiting"),
                Point(1024, 337 + l * 30), .48, WHITE);
        }
        label(ui, "RANDOM RETURN: TOP FIRST", Point(895, 437), .45, WHITE, 2);
        for (int k = 0; k < 3; ++k) {
            int l = 2 - k;
            string target = active >= 0 && a.returnSent[active] ? "P" + to_string(a.returnDest[active][l] + 1) : "await empty slots";
            label(ui, "L" + to_string(l + 1) + "  >  " + target, Point(895, 463 + k * 24), .44, MUTED);
        }
        label(ui, a.serialConnected ? "COM: " + a.comName + " connected" : "COM: OFFLINE", Point(895, 553), .45, a.serialConnected ? ink(GREEN) : GOLD);
        button(ui, Rect(895, 574, 175, 30), "UNUSED ONLY", !a.allowRepeatPickups);
        button(ui, Rect(1078, 574, 175, 30), "ALLOW REPEATS", a.allowRepeatPickups);
        label(ui, "Camera-confirmed cycles: " + to_string(a.completedCycles) + (a.sentCount ? " / mode locked" : ""), Point(895, 625), .40, WHITE);
        string blocked = startBlock(a);
        button(ui, Rect(895, 640, 355, 58), blocked.empty() ? "START AUTOMATIC CYCLE" : a.sentCount ? "CYCLE RUNNING - PLEASE WAIT" : "START UNAVAILABLE", blocked.empty());
        label(ui, "S start    1 / 2 / 3 select    Q exit", Point(895, 723), .40, MUTED);
        panel(ui, Rect(10, 744, 1260, 124));
        if (a.setupOpen) {
            label(ui, "SETUP  /  changes allowed between cycles", Point(25, 774), .48, WHITE, 2);
            const array<string, 9> names = { {"M MAP", "V VERIFY", "R SAMPLE", "G SAMPLE", "B SAMPLE", "P REFRESH", "Z STACK ZONE", "X HISTORY", "K HSV RESET"} };
            for (int i = 0; i < 9; ++i) button(ui, Rect(25 + i * 138, 795, 129, 38), names[i]);
            button(ui, Rect(25, 833, 190, 28), "RESET MAPPING");
            button(ui, Rect(225, 833, 200, 28), "RESET ALL SETTINGS");
            button(ui, Rect(435, 833, 220, 28), a.learnPick || a.learning ? "CANCEL LEARN" : "LEARN 40mm BLOCK");
            string measurement = a.learnedFace.width > 0 ? to_string(cvRound(a.learnedFace.width)) + "x" + to_string(cvRound(a.learnedFace.height)) + " camera px (local reference)" : "No block size saved";
            if (a.learnedFace.width > 0 && a.learnedResolution != a.cameraSize) measurement = "Camera resolution changed - relearn size";
            if (measurement.size() > 48) measurement.resize(48);
            label(ui, measurement, Point(675, 853), .32, MUTED);
            button(ui, Rect(1060, 833, 190, 28), "NEW BATCH");
        }
        else {
            int empty = 0, unsure = 0;
            for (const auto& pos : a.positions) { unsure += pos.uncertain; empty += !pos.uncertain && pos.confirmed == UNKNOWN; }
            label(ui, "NEXT STEP", Point(25, 775), .47, GOLD, 2);
            string next = active < 0 ? (blocked.empty() ? "Choose a pattern, then press START. The return runs automatically." : blocked) :
                !a.returnSent[active] ? "Wait: camera needs three stable empty source slots; robot moves home before receiving return." :
                "Wait: the robot returns L3, L2, L1. Camera checks their colours at the assigned destinations.";
            label(ui, next, Point(25, 805), .43, WHITE);
            if (active >= 0 && !a.returnSent[active] &&
                chrono::duration<double>(chrono::steady_clock::now() - a.stackSentAt[active]).count() >= 30.0) {
                bool noneMoved = true;
                for (int src : a.sentSources[active])
                    if (a.positions[src].uncertain || a.positions[src].confirmed == UNKNOWN) noneMoved = false;
                if (noneMoved) label(ui, "NO PICKUP CHANGE: check RAPID running, RX BYTE log and dg input. No robot acknowledgement.", Point(25, 834), .40, GOLD);
                else label(ui, "Watching pickup changes; robot receipt is not confirmed by COM.", Point(25, 834), .42, MUTED);
            }
            else label(ui, "Empty slots: " + to_string(empty) + "    Uncertain: " + to_string(unsure) + "    Camera updates every frame", Point(25, 834), .45, MUTED);
            label(ui, "Use the physical robot STOP / emergency stop to stop motion.", Point(25, 856), .39, GOLD);
        }

        panel(ui, Rect(10, 876, 1260, 195));
        label(ui, "FIXED PICKUP TARGETS  /  P1-P9 (ABB TAUGHT POSITIONS)", Point(25, 899), .53, WHITE, 2);
        for (int i = 0; i < N; ++i) {
            int col = i % 9, x = 24 + col * 139, y = 916;
            Rect tile(x, y, 131, 131);
            const auto& st = a.positions[i];
            string state = a.mapped != N ? "UNMAPPED" : st.uncertain ? "UNCERTAIN" : st.confirmed == UNKNOWN ? "EMPTY" : cname(st.confirmed);
            Scalar color = state == "UNCERTAIN" ? GOLD : ink(st.confirmed);
            fill(ui, tile, Scalar(43, 49, 61)); rectangle(ui, tile, a.mapped == N ? color : MUTED, 2);
            label(ui, "P" + to_string(i + 1), Point(x + 11, y + 25), .57, WHITE, 2);
            if (state == "RED" || state == "GREEN" || state == "BLUE") fill(ui, Rect(x + 41, y + 37, 49, 40), ink(st.confirmed));
            else rectangle(ui, Rect(x + 41, y + 37, 49, 40), MUTED, 1);
            if (a.processedBlocks[i]) label(ui, a.allowRepeatPickups ? "USED - ALLOWED" : "USED - SKIP", Point(x + 8, y + 126), .32, GOLD, 1);
            label(ui, state, Point(x + 8, y + 107), state.size() > 7 ? .34 : .44, state == "EMPTY" ? MUTED : color, 1);
        }
        fill(ui, Rect(0, 1080, W, 40), Scalar(30, 34, 44));
        string msg = a.sampleMode ? string("CLICK ") + cname(a.sampleColour) + " BLOCK CENTRE TO CALIBRATE" : a.mapping ? "MAP: CLICK P" + to_string(a.mapped + 1) : a.message;
        if (msg.size() > 110) msg.resize(110);
        label(ui, msg, Point(18, 1105), .48, GOLD);
        label(ui, BUILD_TAG, Point(W - 360, 1105), .4, MUTED);
        drawNavigation(ui, a);
        if (preview) { imwrite(a.setupOpen ? "operator_ui_setup_preview.png" : "operator_ui_preview.png", ui); break; }
        imshow("HAND EYE | HD Dashboard", ui);

        int key = a.pendingKey >= 0 ? a.pendingKey : waitKey(25); a.pendingKey = -1;
        if (key < 0) continue;
        if (key == 'o' || key == 'O') {
            if (!a.sentCount) { a.setupOpen = !a.setupOpen; a.mapping = false; a.sampleMode = false; a.zoneSetup = 0; a.learnPick = false; a.learning = false; }
            continue;
        }
        if (a.sentCount && key != 'q' && key != 'Q' && key != 27) continue;
        if (!a.setupOpen && string("nNlLcC0dDtTzZxXmMvVrRgGbBkKpPuU").find(char(key)) != string::npos) continue;
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == 's' || key == 'S') {
            string reason = startBlock(a);
            if (reason.empty()) operatorStart(a); else a.message = reason;
            continue;
        }
        if (key == 'n' || key == 'N') { newBatch(a); continue; }
        if (key == 'l' || key == 'L') {
            if (a.learnPick || a.learning) { a.learnPick = false; a.learning = false; a.sizeSamples.clear(); a.message = "Measurement cancelled"; }
            else {
                a.mapping = false; a.sampleMode = false; a.zoneSetup = 0; a.learnPick = true;
                a.message = "LEARN: place one 40x40mm face flat at pickup height, then click its outline";
            }
            continue;
        }
        if (key == 'c' || key == 'C' || key == '0') { operatorClear(a); continue; }
        if (key == 'd' || key == 'D') { operatorReturn(a); continue; }
        if (key == 't' || key == 'T') { operatorRelease(a); continue; }
        if (key == 'z' || key == 'Z') { a.zoneSetup = -1; a.message = "STACK ZONE: press 1, 2 or 3, then click zone centre"; continue; }
        if (a.zoneSetup == -1 && (key == '1' || key == '2' || key == '3')) { a.zoneSetup = key - '0'; a.message = "Click S" + to_string(a.zoneSetup) + " stack zone centre"; continue; }
        if (key == 'x' || key == 'X') {
            for (int j = 0; j < 3; ++j) { a.observedHistory[j].clear(); a.stackTop[j] = 0; a.topCandidate[j] = 0; a.topStreak[j] = 0; }
            a.message = "Cleared stack observation history";
        }
        if (key == 'm' || key == 'M') {
            if (!a.planLocked) { a.learnPick = false; a.learning = false; a.processedBlocks.fill(false); a.autoProposal = false; a.mapping = true; a.mapped = 0; a.verifiedMap = false; a.autoCandidate = false; resetObservations(a); a.message = "Click ABB P1-P9 in order"; }
        }
        if (key == 'v' || key == 'V') {
            if (a.mapped == N && !a.mapping) { a.verifiedMap = true; saveConfig(a); a.message = "Verified reference saved. Ensure labels match ABB taught P1-P9."; }
        }
        if (key == '1' || key == '2' || key == '3') { a.selected = key - '0'; a.releaseArmed = false; }
        if (key == 'r' || key == 'R' || key == 'g' || key == 'G' || key == 'b' || key == 'B') {
            if (!a.planLocked) { a.sampleColour = (key == 'r' || key == 'R') ? RED : (key == 'g' || key == 'G') ? GREEN : BLUE; a.sampleMode = true; }
        }
        if (key == 'k' || key == 'K') {
            if (!a.planLocked) { a.hue = { {0,0,60,110} }; a.tolerance = { {0,12,15,15} }; a.satMin = 65; a.valMin = 60; resetObservations(a); saveConfig(a); a.message = "HSV defaults restored"; }
        }
        if (key == 'p' || key == 'P') {
            if (a.mapped != N || a.mapping || !a.verifiedMap) { a.message = "Verify P1-P9 against ABB taught positions first (V or M)"; continue; }
            refreshPlans(a);
            a.message = "VISION REFRESHED: choose any unused S, then send one stack";
        }
        if (key == 'u' || key == 'U') {
            a.planLocked = false; a.planRefreshReady = false;
            resetObservations(a);
            a.message = "UI refreshed; reservations retained (use T after robot returns blocks).";
        }
    }
    if (a.port) operatorClear(a);   // final DIO clear; cannot stop a running robot
    disconnect(a); cap.release(); destroyAllWindows();
    return 0;
}
