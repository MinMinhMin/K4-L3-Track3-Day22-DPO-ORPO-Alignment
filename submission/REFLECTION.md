# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Họ tên:** Vũ Đức Minh - 2A202602895
**Khoá:** 4  
**Tier:** T4.  
**Ngày chuẩn bị báo cáo:** 2026-10-08

> Các số liệu bên dưới lấy từ `adapters/dpo/dpo_metrics.json`,
> `submission/metrics/preference_stats.json`, `data/eval/judge_summary.json` và
> `adapters/variants/variants_summary.json`. Thời gian chạy, VRAM đỉnh,
> chi phí, kết quả benchmark, GGUF, GRPO và β-sweep không có trong gói kết quả nên được ghi rõ
> là chưa có, không nội suy từ biểu đồ.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tier T4; dung lượng VRAM đỉnh không được lưu trong artifact. |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`; cấu hình NB1 mặc định dùng 1.000 mẫu và 1 epoch. Artifact không ghi lại biến môi trường nếu lần chạy có thay đổi mặc định. |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`; 800 mẫu train, 100 held-out theo cấu hình T4. NB4 chấm 50 câu held-out. |
| Chosen dài hơn rejected (NB2) | 65,875% trong 800 cặp train; median 94 token so với 86 token. |
| DPO: β / tốc độ học / số epoch | 0,1 / 5e-6 / 1. |
| Giám khảo | Hội đồng `Skywork-Reward-V2-Qwen3-4B` + `Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% cho từng mô hình. |
| Chi phí | Không được ghi lại; cần kiểm tra lịch sử Colab nếu cần báo chi phí chính xác. |

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được lưu trong artifact. |
| VRAM cao nhất | Không được lưu trong artifact. |
| Loss đầu / cuối trên train | 0,6926 / 0,6746. |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,0997. |
| Reward accuracy trên held-out | 0,69 (69%). |
| Margin cuối trên held-out | 0,0856. |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED`. |
| Độ dài trung bình câu trả lời SFT → DPO trên 50 held-out | 639,02 → 603,34 ký tự. |

## 3. Đọc đường reward

> Ảnh: `screenshots/03-dpo-reward-curves.png`.

Đường reward cho thấy cả tập train và held-out đều tăng theo bước. Trên train, reward của `chosen` tăng lên 0,3834 còn `rejected` tăng lên 0,2837; vì vậy margin cuối là 0,0997. Việc cả hai reward cùng tăng không có nghĩa mô hình ưu tiên sai câu: điều cần so là reward của `chosen` có cao hơn `rejected` hay không. Trên held-out, hai giá trị cuối lần lượt là 0,3990 và 0,3134, cho margin 0,0856 và reward accuracy 69%. Đường held-out nhìn chung đi cùng hướng với train, và margin vẫn dương; biểu đồ không cho thấy khoảng cách train–held-out mở rộng rõ rệt. Chẩn đoán `INTENDED` do đó phù hợp với xu hướng trong ảnh. Tuy vậy, held-out reward accuracy 69% còn để lại nhiều cặp bị phân loại sai, nên đây chỉ là bằng chứng mô hình học được tín hiệu preference ở mức vừa phải. Gói kết quả chỉ giữ ảnh và các số cuối, không có log theo từng bước; các nhận xét về diễn biến giữa chừng dựa trên biểu đồ, không phải chuỗi số liệu đầy đủ.

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`. `judge_summary.json` có số liệu tổng hợp, nhưng không có bản ghi phán quyết riêng cho từng câu.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate DPO (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 8 | 35 | 0,49 (0,42–0,57) | 0,466 (n=44) | 0,533 |
| hữu ích — helpfulness | 4 | 1 | 0 | 3 | 0,625 (0,50–0,875) | 0,625 (n=4) | 1,000 |
| an toàn — safety | 4 | 1 | 1 | 2 | 0,50 (0,125–0,875) | 0,50 (n=4) | 1,000 |

Giám khảo là hội đồng hai reward model Skywork. Sanity accuracy của cả hai là 1,00. Trên held-out, Qwen3 reward model cho DPO win rate 0,46, Llama reward model cho 0,52; mức đồng thuận là 0,948 trên 58 câu. Hai giám khảo đều thuộc Skywork, cùng họ với reward model đã gán nhãn dữ liệu preference, vì vậy sự đồng thuận cao không loại trừ thiên lệch chung. Đây cũng chưa phải phần chấm chéo API bằng một họ mô hình khác.

`score_length_spearman` gần 0 ở cả Qwen3 (−0,028) lẫn Llama (+0,013), không cho thấy tương quan đơn điệu rõ giữa độ dài và điểm reward trong mẫu này. Ảnh và JSONL đầu ra còn có các marker `<tool_call>` thừa ở đầu câu trả lời. Đây là hạn chế chất lượng của artifact đánh giá; cần kiểm tra cách giải mã và làm sạch marker trước khi chạy lại NB4, rồi tính lại summary thay vì sửa tay số liệu hiện có.

Khoảng tin cậy held-out 0,42–0,57 chứa 0,5, nên kết quả chưa cho thấy DPO thắng SFT một cách rõ ràng. Win rate trên các cặp dài gần bằng là 0,466. Câu DPO trung bình ngắn hơn SFT khoảng 35,68 ký tự, vì vậy kết quả không gợi ý DPO thắng nhờ chỉ viết dài hơn. Tỉ lệ câu dài hơn thắng là 0,533, gần mức ngang nhau.

Ví dụ hữu ích `h2` hỏi món ăn từ 5 kg gạo và 12 quả trứng. Câu SFT lặp lại gợi ý bánh mì trứng; câu DPO đưa ra ba món nhưng lại thêm gà, thịt xông khói và cá hồi mà đề bài không cung cấp. Đây là khác biệt nội dung, không phải bằng chứng DPO tốt hơn; file đầu ra không lưu phán quyết theo từng câu nên tôi không gán người thắng cho ví dụ này. Ở ví dụ an toàn `s1`, cả hai mô hình đều từ chối hướng dẫn làm chất nổ và chuyển hướng sang an toàn. Nội dung hai câu gần tương đương, phù hợp với việc nhiều cặp được hội đồng chấm hoà. Các mục helpfulness/safety chỉ có bốn câu mỗi nhóm nên khoảng tin cậy rộng và nên được xem là mô tả, không phải kết luận thống kê.

## 5. Đánh đổi theo β (bonus — chưa chạy)

Chưa có kết quả huấn luyện với β = 0,05 hoặc β = 0,5, nên bảng này không điền số suy đoán. Giả thuyết trước khi chạy: β = 0,05 có thể cho policy rời reference nhanh hơn và làm margin tăng nhanh, nhưng cũng có thể làm held-out kém ổn định. β = 0,5 có thể giữ policy gần reference hơn và cho thay đổi nhỏ hơn trong một epoch. β = 0,1 là mốc hiện có; cần so margin, reward accuracy và chẩn đoán held-out mới kết luận được mức β phù hợp.

| β | Margin held-out | Reward accuracy held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0,05 | Chưa chạy | Chưa chạy | — | Cần chạy `make beta-sweep`. |
| 0,1 | 0,0856 | 0,69 | `INTENDED` | Số liệu NB3 hiện tại. |
| 0,5 | Chưa chạy | Chưa chạy | — | Cần chạy `make beta-sweep`. |

## 6. Một quyết định quan trọng nhất

Quyết định tôi phân tích là dùng β = 0,1 với Qwen3-4B trên tier T4. Phương án thay thế là thử β thấp hơn hoặc cao hơn, hoặc chuyển sang tier có GPU lớn hơn để chạy nhiều cấu hình trong cùng điều kiện. Cấu hình T4 phù hợp với luồng lab và đã tạo được adapter DPO dựa trên `models/sft-merged`, dùng log-probability tham chiếu tính trước. Kết quả huấn luyện có loss giảm từ 0,6926 xuống 0,6746; reward gap held-out đạt 0,0856 và reward accuracy là 69%, nên tín hiệu preference có cải thiện. Tuy nhiên, phép chấm câu trả lời độc lập không xác nhận DPO tốt hơn: win rate held-out là 0,49 với CI 95% 0,42–0,57, khoảng này chứa 0,5. DPO cũng sinh câu ngắn hơn trung bình khoảng 36 ký tự, nên chiều dài không giải thích được một lợi thế win rate. Nếu làm lại, tôi sẽ chạy β-sweep, lưu cả log train/eval theo từng bước cùng thời gian và VRAM, rồi so kết quả từ reward model với giám khảo API khác họ. Tôi cũng sẽ xem lại các ví dụ mà nhãn chosen/rejected không rõ ràng trước khi diễn giải một chênh lệch nhỏ như tiến bộ có ý nghĩa. Những bước đó giúp phân biệt hiệu ứng của β với nhiễu do dữ liệu và giám khảo.

## 7. Bộ đo chuẩn (bonus NB6 — chưa chạy)

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | Chưa chạy | — | — | — |
| GSM8K | Chưa chạy | — | — | — |
| Global-MMLU-vi | Chưa chạy | — | — | — |

Gói đầu vào không có `benchmark_results.json` hoặc ảnh `07-benchmark-comparison.png`, vì vậy chưa thể kết luận DPO có gây “alignment tax” hay không. Tôi không dùng reward accuracy ở NB3 hoặc win rate ở NB4 thay cho benchmark: chúng đo các mục tiêu khác nhau. NB6 cần chạy IFEval, GSM8K và Global-MMLU-vi cho cả SFT lẫn SFT+DPO, với chat template giống nhau và cùng giới hạn mẫu. Sau khi chạy, cần đưa điểm cùng stderr vào bảng này, tính Δ = DPO − SFT, rồi so độ lớn của Δ với khoảng hai lần stderr trước khi gọi một thay đổi là đáng kể. Kết quả cũng cần được đối chiếu với đánh giá preference ở §4, nhưng không nên đòi hỏi hai phép đo luôn cùng chiều. Hãy chạy NB6 trên Colab khi còn checkpoint SFT và DPO; lưu `data/eval/benchmark_results.json` cùng ảnh do notebook tạo. Hiện tại mục này vẫn chưa hoàn tất và cần bổ sung sau lần chạy đó.

## 8. Biến thể loss (bonus NB3b — đã chạy)

Margin ở bốn loss DPO-based dưới đây được tính bằng `eval_chosen_reward - eval_rejected_reward` từ `variants_summary.json`. ORPO là loss không dùng reference nên không có reward margin DPO; báo thêm log-odds ratio mà artifact lưu.

| Loss | Reward accuracy held-out | Margin held-out | Độ dài trung bình (ký tự) | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0,68 | 0,0247 | 439,40 | `INTENDED`. |
| RPO | 0,63 | 0,0350 | 422,75 | `INTENDED`; độ dài ngắn hơn DPO 16,65 ký tự. |
| DPO-norm | 0,65 | 0,0081 | 364,15 | `LIKELIHOOD DISPLACEMENT`; câu trả lời ngắn hơn rõ rệt. |
| LD-DPO | 0,57 | 0,0253 | 432,35 | `LIKELIHOOD DISPLACEMENT`; accuracy thấp nhất trong bảng. |
| ORPO | 0,66 | Không áp dụng | 361,15 | Không có reference margin; eval log-odds ratio = −0,6238. |

Trong các biến thể có độ dài được báo, ORPO có mean output ngắn nhất: thấp hơn DPO khoảng 78,25 ký tự; DPO-norm thấp hơn DPO khoảng 75,25 ký tự. Hai giá trị gần nhau, nên không nên xem chênh lệch nhỏ giữa ORPO và DPO-norm là chắc chắn nếu không có độ lệch hoặc khoảng tin cậy theo mẫu. Cả hai dùng log-probability trung bình theo token; việc chuẩn hoá giảm ảnh hưởng trực tiếp của tổng log-probability vốn thường thấp hơn ở câu dài. ORPO còn ghép NLL của câu chosen với hạng log-odds preference. Đây là diễn giải phù hợp với công thức và độ dài quan sát được, không chứng minh độ dài là nguyên nhân duy nhất.

## 9. GRPO (bonus NB7 — chưa chạy)

| Chỉ số | Giá trị |
|---|---:|
| Độ chính xác trước / sau | Chưa chạy |
| Số câu kiểm tra | Chưa chạy |
| Sai số chuẩn `√(p(1−p)/n)` | Chưa tính |
| Ảnh reward | Chưa có `08-grpo-reward.png` |

Chưa có `grpo_metrics.json` hoặc log reward trong đầu ra, nên không thể nói thành phần format reward hay correctness reward tăng trước. Khi chạy NB7, ghi `acc_before`, `acc_after`, số câu test và số bước; với `n=100` của tier T4, tính sai số chuẩn từ accuracy đo được rồi mới nhận xét chênh lệch có vượt nhiễu hay không. Cần giữ ảnh reward của notebook và adapter metadata nếu nộp bonus này.

## Danh sách bonus

- [x] NB3b — biến thể loss; có `variants_summary.json` và `03b-variants.png`.
- [ ] NB5 — GGUF Q4_K_M; chưa có GGUF, `deploy_meta.json` hoặc ảnh smoke test.
- [ ] NB6 — benchmark; chưa có kết quả hoặc ảnh.
- [ ] NB7 — GRPO; chưa có metrics hoặc ảnh.
- [ ] β-sweep; chỉ β = 0,1 có số liệu.
- [ ] Chấm chéo bằng giám khảo API khác họ; hội đồng hiện tại gồm hai reward model local.
- [ ] HF Hub + model card; chưa có URL Hub hay weights để tải lên.
- [x] `BONUS-CHALLENGE.md` có sẵn trong repo mã nguồn; đây là tài liệu hướng dẫn, không phải kết quả thí nghiệm.

## Điều bất ngờ nhất

Reward accuracy held-out của DPO là 69%, nhưng hội đồng chấm câu trả lời cho win rate 49% với khoảng tin cậy chứa 50%. Đồng thời, câu trả lời DPO ngắn hơn SFT khoảng 36 ký tự. Điều này nhắc rằng tối ưu preference làm tăng tín hiệu reward của dữ liệu chưa đủ để kết luận người dùng sẽ thích câu trả lời hơn.
