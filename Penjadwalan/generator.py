import re
import torch

def convert_duration_to_weeks(duration_str):
    match = re.match(r"(\d+)\s*(minggu|bulan)", duration_str.lower())
    if not match:
        raise ValueError(f"Format durasi tidak dikenali: {duration_str}. Gunakan format seperti 'X minggu' atau 'Y bulan'.")
    jumlah, satuan = int(match.group(1)), match.group(2)
    return jumlah if satuan == "minggu" else jumlah * 4

def calculate_bmi(weight, height):
    if not isinstance(weight, (int, float)) or not isinstance(height, (int, float)):
        return 0.0, "Tidak Valid", "Berat dan tinggi badan harus berupa angka."
    if height <= 0:
        return 0.0, "Tidak Diketahui", "Tinggi badan harus merupakan angka positif lebih dari 0."
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    if bmi < 18.5:
        category = "Kurus (Underweight)"
        suggestion = "Disarankan untuk menambah asupan kalori dari makanan bergizi dan konsultasi dengan ahli gizi."
    elif bmi < 24.9:
        category = "Normal (Ideal)"
        suggestion = "Pertahankan pola makan seimbang dan aktivitas fisik teratur."
    elif bmi < 29.9:
        category = "Kelebihan Berat Badan (Overweight)"
        suggestion = "Disarankan untuk mengurangi asupan makanan tinggi lemak dan gula, serta meningkatkan aktivitas fisik."
    else:
        category = "Obesitas"
        suggestion = "Sangat disarankan untuk berkonsultasi dengan dokter atau ahli gizi untuk program penurunan berat badan yang aman."
    return round(bmi, 2), category, suggestion

def generate_week_prompt(goal, duration, age, weight, height, eatingPattern,
                         allergies, dislikes, exerciseFrequency, sleepQuality,
                         week_num, start_day, end_day, bmi, bmi_category, bmi_suggestion):
    return f"""
Buat program sesuai dengan tujuan {goal} untuk {duration} minggu. Fokus pada detail jadwal harian untuk Minggu Ke-{week_num} (Hari {start_day}-{end_day}).

Harap sediakan konten untuk bagian-bagian berikut:
Breakfast (07:00-08:00)
Lunch (12:00-13:00)
Dinner (18:00-19:00)
Snack (15:30)
Exercise
Tips Harian secara singkat.

Untuk setiap item makanan atau exercise, berikan: Menu/Jenis, Porsi, Kalori, dan Catatan singkat untuk rekomendasi apasaja yang harus dikonsumsi dan dilakukan.
Untuk Tips Harian dan Saran Olahraga, berikan poin-poin. Pastikan format output nya sama tiap minggunya.

Informasi pengguna untuk dipertimbangkan:
- Tujuan: {goal}
- Usia: {age} tahun
- Berat badan saat ini: {weight} kg
- Tinggi badan: {height} cm
- BMI saat ini: {bmi}, Kategori: {bmi_category} (Saran: {bmi_suggestion})
- Pola makan umum: {eatingPattern}
- Alergi makanan: {allergies}
- Makanan tidak disukai: {dislikes}
- Frekuensi olahraga: {exerciseFrequency}
- Kualitas tidur: {sleepQuality}

Hindari makanan yang menyebabkan alergi ({allergies}) dan yang tidak disukai ({dislikes}).
Pastikan saran realistis dan sesuai dengan konteks Indonesia.
Jangan sertakan pengulangan instruksi atau informasi BMI secara eksplisit dalam output jadwal harian Anda.
""".strip()

def generate_diet_program(user_info, tokenizer, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    try:
        total_weeks = convert_duration_to_weeks(user_info.get('duration', '1 minggu'))
    except ValueError as e:
        return f"Error pada durasi: {e}"

    bmi, category, suggestion = calculate_bmi(
        user_info.get('weight', 0), user_info.get('height', 0)
    )
    if category == "Tidak Valid":
        return f"Error pada input pengguna: {suggestion}"

    hasil = []

    hasil.append(f"# Program {user_info.get('goal', 'Diet')} {user_info.get('duration')} - Fokus Sehat\n")
    hasil.append("## Tujuan")
    hasil.append(f"Program ini dirancang untuk membantu Anda mencapai tujuan '{user_info.get('goal')}' secara sehat dan berkelanjutan selama {user_info.get('duration')}. Program ini mempertimbangkan alergi makanan ({user_info.get('allergies')}) dan makanan yang tidak disukai ({user_info.get('dislikes')}).\n")
    hasil.append("## BMI dan Kategori")
    hasil.append(f"**BMI:** {bmi} ({category})\n")
    hasil.append("## Durasi")
    hasil.append(f"{user_info.get('duration')} ({total_weeks * 7} hari)\n")
    hasil.append("## Jadwal Harian\n")

    for week in range(1, total_weeks + 1):
        start_day, end_day = (week - 1) * 7 + 1, week * 7
        prompt = generate_week_prompt(
            goal=user_info.get('goal', 'Tidak spesifik'),
            duration=user_info.get('duration', f'{total_weeks} minggu'),
            age=user_info.get('age', 0),
            weight=user_info.get('weight', 0),
            height=user_info.get('height', 0),
            eatingPattern=user_info.get('eatingPattern', 'Tidak spesifik'),
            allergies=user_info.get('allergies', 'Tidak ada'),
            dislikes=user_info.get('dislikes', 'Tidak ada'),
            exerciseFrequency=user_info.get('exerciseFrequency', 'Tidak spesifik'),
            sleepQuality=user_info.get('sleepQuality', 'Tidak spesifik'),
            week_num=week,
            start_day=start_day,
            end_day=end_day,
            bmi=bmi,
            bmi_category=category,
            bmi_suggestion=suggestion
        )
        
        messages = [
            {"role": "system", "content": "Anda adalah ahli gizi dan nutrisi berpengalaman di Indonesia. Buat rencana diet mingguan sesuai konteks lokal."},
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=2000, do_sample=False)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded.split(prompt)[-1].strip()

        # hasil.append(f"### Hari {start_day}-{end_day} (Minggu {'Pertama' if week == 1 else 'Kedua' if week == 2 else 'Ketiga' if week == 3 else f'Ke-{week}'})\n")
        hasil.append(response_text.strip() + "\n")

    return "\n".join(hasil)
