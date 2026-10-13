 library(reticulate)
  repl_python()
  


import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ==========================================
# 1. تحميل وتجهيز البيانات
# ==========================================
# تحميل البيانات النظيفة المحفوظة مسبقاً
total_trips_clean = pd.read_csv("my_final_clean_data.csv")

# تحويل التواريخ واستخراج الميزات
total_trips_clean['started_at'] = pd.to_datetime(total_trips_clean['started_at'], format='mixed')
total_trips_clean['month'] = total_trips_clean['started_at'].dt.month.astype(str) # تحويل الشهر إلى نص ليتعامل معه OneHotEncoder كفئة

# أخذ عينة عشوائية لتسريع التدريب
df_sample = total_trips_clean.sample(n=100000, random_state=42)

# تحويل المتغير المستهدف إلى أرقام (1 للأعضاء، 0 للزوار)
df_sample['user_type'] = df_sample['member_casual'].map({'member': 1, 'casual': 0})

# تحديد المدخلات (X) والهدف (y)
# سنأخذ الميزات العددية والشهور (كمتغير تصنيفي Categorical)
X = df_sample[['ride_length', 'started_hour', 'month']]
y = df_sample['user_type']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. بناء خط الأنابيب (Pipeline) مع ColumnTransformer
# ==========================================
# معالجة الأعمدة التصنيفية (تطبيق OneHotEncoder) وترك الأعمدة العددية كما هي (passthrough)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['month'])
    ],
    remainder='passthrough' # الحفاظ على الأعمدة الأخرى (ride_length, started_hour) بدون تعديل
)

# دمج معالجة البيانات مع نموذج Random Forest في Pipeline واحد
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# ==========================================
# 3. تدريب النموذج وتقييمه
# ==========================================
print("جاري تدريب النموذج...")
full_pipeline.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred = full_pipeline.predict(X_test)

print(f"\nدقة النموذج الإجمالية: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred, target_names=['Casual', 'Member']))

# ==========================================
# 4. حفظ النموذج واسترجاعه
# ==========================================
joblib.dump(full_pipeline, 'my_smart_pipeline_model.pkl')
print("\nتم حفظ النموذج بنجاح باسم 'my_smart_pipeline_model.pkl'")

# استرجاع النموذج للتأكد
loaded_pipeline = joblib.load('my_smart_pipeline_model.pkl')

# ==========================================
# 5. تجربة التنبؤ بصف جديد (بكل سهولة وبدون تعقيد!)
# ==========================================
# لاحظ أننا ندخل البيانات الخام مباشرة دون أي عمليات تحويل يدوية معقدة
new_row = pd.DataFrame([{
    'ride_length': 64.3,
    'started_hour': 17,
    'month': '12'  # الشهر كـ نص
}])

prediction = loaded_pipeline.predict(new_row)
user_status = "Member (عضو)" if prediction[0] == 1 else "Casual (زائر عشوائي)"
print(f"\nالنتيجة المتوقعة لهذا الصف الجديد هي: {user_status}")
#======================================

