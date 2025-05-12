from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
#yyyy-MM-dd HH:mm:ss
curdatetime =datetime.now().replace(microsecond=0)
print(curdatetime)#현재날짜
pre_datetime=curdatetime-timedelta(days=200)
print(pre_datetime)
ppre_datetime=pre_datetime-timedelta(hours=240)
print(ppre_datetime)
pppre_datetime=ppre_datetime-timedelta(minutes=1*10)
print(pppre_datetime)
pre_week = pppre_datetime-relativedelta(weeks=500)
print(pre_week)
pre_month = pre_week-relativedelta(months=500)
print(pre_month)


