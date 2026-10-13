import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. تحميل وتجهيز البيانات
# ==========================================
total_trips_clean = pd.read_csv("my_final_clean_data.csv")

# تحويل عمود التاريخ
total_trips_clean['started_at'] = pd.to_datetime(
    total_trips_clean['started_at'], format='mixed'
)
total_trips_clean['month'] = total_trips_clean['started_at'].dt.month

# أخذ العينة العشوائية
df_sample = total_trips_clean.sample(n=100000, random_state=42)

# تحويل الفئات المستهدفة إلى أرقام: Member = 1, Casual = 0
df_sample['user_type'] = df_sample['member_casual'].map({'member': 1, 'casual': 0})

# تجهيز ميزات التدريب (X) والهدف (y)
X = df_sample[['ride_length', 'started_hour']] 
# استخدام drop_first=True لتجنب الارتباط الخطي (تأكد أن الأعمدة ستكون من 2 إلى 12)
months_dummies = pd.get_dummies(df_sample['month'], drop_first=True)
X = pd.concat([X, months_dummies], axis=1)

y = df_sample['user_type']

# توحيد أسماء الأعمدة كنصوص
X.columns = X.columns.astype(str)

# حفظ قائمة أعمدة التدريب لضمان تطابقها لاحقاً أثناء التنبؤ اليدوي
training_columns = X.columns.tolist()

# ==========================================
# 2. تقسيم البيانات والتدريب
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ واختبار الدقة
y_pred = model.predict(X_test)
print(f"دقة النموذج الإجمالية: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred, target_names=['Casual', 'Member']))

# ==========================================
# 3. حفظ النموذج واسترجاعه
# ==========================================
joblib.dump(model, 'my_model.pkl')
print("تم حفظ النموذج بنجاح باسم 'my_model.pkl'")

loaded_model = joblib.load('my_model.pkl')

# ==========================================
# 4. اختبار صف جديد (بالطريقة اليدوية الصحيحة 100%)
# ==========================================
# إدخال الصف الخام
raw_row = pd.DataFrame([{
    'ride_id': '1639A96F29FF2E62',
    'rideable_type': 'electric_bike',
    'started_at': '2021-09-14 10:18:52',
    'ended_at': '2021-09-14 10:18:58',
    'start_station_name': 'Dearborn St & Monroe St',
    'start_station_id': 'TA1305000006',
    'end_station_name': 'Dearborn St & Monroe St',
    'end_station_id': 'TA1305000006',
    'start_lat': 41.881189,
    'start_lng': -87.6297916666666,
    'end_lat': 41.88115,
    'end_lng': -87.62970
}])

# استخراج الميزات تلقائياً
raw_row['started_at'] = pd.to_datetime(raw_row['started_at'])
raw_row['ended_at'] = pd.to_datetime(raw_row['ended_at'])

raw_row['ride_length'] = (raw_row['ended_at'] - raw_row['started_at']).dt.total_seconds() / 60.0
raw_row['started_hour'] = raw_row['started_at'].dt.hour
month_val = raw_row['started_at'].dt.month.values[0]

# بناء هيكل الأعمدة المطابق تماماً لـ X_train باستخدام قائمة الأعمدة المحفوظة
processed_row = pd.DataFrame(0, index=[0], columns=training_columns)

# تعبئة القيم الأساسية
processed_row['ride_length'] = raw_row['ride_length']
processed_row['started_hour'] = raw_row['started_hour']

# تعبئة عمود الشهر المناسب بـ 1 (إذا كان موجوداً ضمن الأعمدة بعد drop_first)
month_col_name = str(month_val)
if month_col_name in processed_row.columns:
    processed_row[month_col_name] = 1

# توحيد أسماء الأعمدة كنصوص والتأكد من ترتيبها التام
processed_row.columns = processed_row.columns.astype(str)
processed_row = processed_row[training_columns]

# التنبؤ بالنتيجة باستخدام النموذج المحمل
prediction = loaded_model.predict(processed_row)
user_status = "Member (عضو)" if prediction[0] == 1 else "Casual (زائر عشوائي)"
print(f"النتيجة المتوقعة لهذا الصف هي: {user_status}")
