# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1:
- `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về một vật thể "cab" của ảnh "trafic", vì mình có thể thấy đây là thông tin mỗi class của một sample trong report.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Đi kèm taxonomy/checkpoint.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Giống như một khu đô thị có tên toà nhà, khu vực, tên đường, tên toà, tên căn hộ, tầng, phòng; thì việc này để ID và chỉ đích danh object đó.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Ta sẽ cần quy định chọn lớp theo đối tượng/chủ đề chính, tiêu chí ưu tiên.
6. Vì sao không được lấy `score` của model làm ground truth?
- Đó chỉ là prediction, cần được con người gán nhãn và kiểm tra, đơn giản là máy sẽ chưa thể hiểu rõ vật thể ngoài đời bằng con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
- `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68.
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên trái: (385.33, 69.24) ; Góc dưới phải: (498.92, 348.92) .
- Rộng = 113.58 ; Cao =279.68.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- 0.20: 17 ; 0.60: 6.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Kiểm tra thêm các hộp có score thấp, đối chiếu lớp với vật thể trong ảnh, tìm hộp trùng hoặc sai vị trí và loại prediction không có đủ bằng chứng. 
- Nhìn ảnh để kiểm tra vật thể bị bỏ sót; không chỉ duyệt những hộp model đã vẽ.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Vẽ hình chữ nhật nhỏ nhất bao kín phần nhìn thấy của từng đối tượng cần gán nhãn, giữ tọa độ trong ảnh và không tự kéo hộp ra ngoài khung để đoán phần bị cắt. Mỗi đối tượng có hộp riêng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật bị che, cắt mép:
"class_id": 41,
    "class_name": "cup",
    "score": 0.381542,
- Nếu chưa quy định mức nhìn thấy tối thiểu hoặc có gán nhãn riêng phần tay hay không, cần escalation lên Lab Coach, không tự suy ra cơ thể ngoài ảnh hay coi prediction person là đáp án chuẩn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
- `instance_id`=traffic-001 ; `class_name`=bus ; `score`=0.925745 ; số điểm polygon=`polygon_point_count`=120 ; 3–5 điểm đầu của `polygon_xy`= 
[148.0,189.0],
[147.0,90.0],
[145.0,190.0],
[143.0,192.0],
[142.0,192.0].
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Box chỉ cho biết phạm vi hình chữ nhật bao đối tượng, còn polygon mô tả biên và hình dạng của vùng đối tượng. Ở người đứng phía phải, polygon có thể bám theo đầu, tay, thân và chân, giúp phân biệt vùng người với nền quanh người hoặc giữa hai chân; box không thể biểu diễn chi tiết đó.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` dùng để xác định một vật thể duy nhất để phân biệt chúng với các đổi tượng cùng class khác trong ảnh.
- Khác với class và tracking ID, `instance_id` dùng để phục vụ việc phân tách các đối tượng độc lập trong một bức ảnh tĩnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Với guideline gán nhãn phần nhìn thấy, đặt polygon theo biên nhìn thấy của từng đối tượng, loại nền và phần thuộc đối tượng khác khỏi mask, không tự tô thêm vùng bị che; các instance cùng lớp vẫn có mask riêng. Nếu công cụ và định dạng có hỗ trợ lỗ/vùng rời thì xử lý theo guideline của dự án; không tự chọn cách nối qua vùng nền khi chưa có quy tắc.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- trên ảnh kitchen, bàn tiền cảnh tiếp xúc và bị che bởi các bát/khay cùng đồ vật đặt trên bàn. Record kitchen-005 được model gọi là dining table, score = 0.578416, với 598 điểm polygon. Cần kiểm tra vùng tiếp giáp ở mép các đồ vật và xác định mask của bàn bao phần nào. Mình escalate vì chưa rõ quầy bếp có thuộc lớp dining table hay không, hoặc có vẽ phần mặt bàn bị đồ vật che hay không, cần escalation; không biến mask của model thành quy tắc chuẩn.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi): 

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Trong ví dụ single-label, một lớp từ taxonomy quy định cho toàn ảnh. | Ảnh traffic có nhiều loại xe nhưng top-1 là cab, score 0.510915; prediction không xác định chiếc xe nào và chưa đủ để xác nhận nhãn toàn ảnh. | Đọc tiêu chí chọn chủ đề chính; chọn một lớp theo guideline hoặc escalation nếu không đủ thông tin. | Kiểm tra nhãn có phù hợp toàn ảnh và quy tắc ưu tiên không; đối chiếu ID/tên/taxonomy, không duyệt chỉ vì top-1 có score cao nhất. |
| Phát hiện vật thể | Một lớp và một bounding box cho mỗi đối tượng cần gán nhãn theo guideline. | kitchen: record person score 0.610931, box [0.12, 263.18, 61.55, 310.78] nằm sát mép trái, chỉ thấy một phần người/tay. | Áp dụng quy tắc đối tượng cắt mép và phần nhìn thấy; không đoán phần ngoài ảnh; hỏi Lab Coach khi thiếu quy tắc. | Kiểm tra chính sách có gán nhãn đối tượng này không, box có bám đúng phần cần gán không, có bỏ sót/nhầm lớp/trùng hộp không; yêu cầu sửa nếu không khớp. |
| Instance segmentation | Một lớp và một mask/polygon riêng cho từng instance theo phạm vi gán nhãn đã quy định. | kitchen-005 được dự đoán dining table, score 0.578416, polygon 598 điểm; bàn/quầy bị đồ vật tiền cảnh che, khó xác định phạm vi lớp và vùng biên. | Tách biên bàn với đồ vật theo guideline; không tự điền vùng bị che; escalation nếu taxonomy hoặc quy tắc biên chưa rõ. | So mask với ảnh, tìm vùng nền/đồ vật khác bị tô lẫn, vùng thiếu và instance bị gộp; kiểm tra lớp và yêu cầu rework. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Mình make a copy của file gg colab, kết nối với cloud, github, làm REPORT.md ở local.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Lab coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [✔️] `REPORT.md` (file này đã điền)
- [✔️] `day1_lab_outputs/classification_predictions.json`
- [✔️] `day1_lab_outputs/detection_predictions.json`
- [✔️] `day1_lab_outputs/segmentation_predictions.json`
- [✔️] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [✔️] `day1_lab_outputs/visuals/classification_top5.png`
- [✔️] `day1_lab_outputs/visuals/detection_predictions.png`
- [✔️] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [✔️] Ô validation cuối notebook báo `PASS`
- [✔️] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [✔️] Đã nộp **link repository** trên LMS
