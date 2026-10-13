# ==========================================
# 0. استيراد المكتبات الأساسية
# ==========================================
library(tidyverse)
library(lubridate)
library(janitor)

# تحديد المسار)
setwd("C:/Users/ayaat/Desktop/cyclingpro")

# ==========================================
# 1. قراءة ودمج الملفات بكفاءة عالية
# ==========================================
trip_files <- list.files(pattern = "*.csv", full.names = TRUE)
total_trips <- map(trip_files, read_csv) |> list_rbind()

# ==========================================
# 2. تنظيف وهندسة البيانات (Data Cleaning & Feature Engineering)
# ==========================================
total_trips_clean <- total_trips %>%
  clean_names() %>% # توحيد وتنسيق أسماء الأعمدة تلقائياً
  mutate(
    started_at = ymd_hms(started_at),
    ended_at = ymd_hms(ended_at)
  ) %>%
  drop_na(start_station_name, end_station_name) %>%
  distinct() %>%
  mutate(
    ride_length = as.numeric(difftime(ended_at, started_at, units = "mins")),
    started_hour = hour(started_at),
    day_of_week = wday(started_at, label = TRUE, abbr = FALSE)
  ) %>%
  filter(
    ride_length > 1, 
    !str_detect(start_station_name, "Quality|HR|base")
  )

# ==========================================
# 3. تحليل البيانات (Aggregation & Analysis)
# ==========================================
# دالة لحساب المنوال (أكثر ساعة تكراراً للرحلات)
get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

User_analysis <- total_trips_clean %>%
  mutate(
    hour = hour(started_at),
    month = month(started_at, label = TRUE)
  ) %>%
  group_by(member_casual, day_of_week, month) %>%
  summarise(
    total_rides = n(),
    average_duration = mean(ride_length, na.rm = TRUE),
    most_frequent_hour = get_mode(hour),
    .groups = "drop"
  ) %>%
  arrange(desc(total_rides))

# طباعة عينة من النتائج للتأكد
print(head(User_analysis))

# ==========================================
# 4. تصدير وحفظ النتائج النهائية
# ==========================================
# استخدام write_csv السريعة والحديثة
write_csv(User_analysis, "my_final_data.csv")
write_csv(total_trips_clean, "my_final_clean_data.csv")

# ==========================================
# 5. الانتقال إلى بيئة Python (عبر reticulate)
# ==========================================
library(reticulate)
repl_python()

#str(total_trips_clean)

# ggplot(total_trips_clean, aes(x = day_of_week, fill = member_casual)) +
#   geom_bar(position = "dodge") +
#   scale_fill_manual(values = c("member" = "steelblue", "casual" = "darkorange")) +
#   labs(
#     title = "Number of trips weekly",
#     x = "Day",
#     y = "Number of trips",
#     fill = "User type"
#   )



 
# 
# library(scales)
# ggplot(User_analysis, aes(x = day_of_week, y = total_rides, fill = member_casual)) +
#   geom_col(position = "dodge") +
#   scale_y_continuous(labels = comma) +
#   scale_fill_manual(values = c("member" = "steelblue", "casual" = "darkorange")) + # أضفتها لك لتوحيد الألوان مع الرسم الأول
#   labs(
#     title = "Number of trips weekly",
#     x = "Day",
#     y = "Number of trips",
#     fill = "User type"
#   )
