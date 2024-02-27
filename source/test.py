# df 전처리
import numpy as np
import pandas as pd

# 시게열 예측
from prophet import Prophet
import datetime as dt

# plotly 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# stramlit
import streamlit as st

# read_html 인증 오류
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 한글깨짐 방지
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False
f_path = "font/KOPUBWORLD DOTUM BOLD.TTF"
font_name = font_manager.FontProperties(fname=f_path).get_name()
rc('font', family=font_name)

# 시계열 데이터프레임 생성 함수
def gen_timeseries_df(n):
  # pd.read_html : 크롤링
  html='https://www.busanpa.com/kor/Contents.do?mCode=MN0931'
  df_raw=pd.read_html(html,encoding='utf-8')[0]
  
  # 멀티컬럼 삭제
  df=df_raw[['연도','총 계']].droplevel(axis=1,level=0)

  # future dataframe 생성
  # make_future_dataframe은 일 단위로만 생성 가능하기 때문에 시계열 예측에 적적하지 않아서 임의로 생성
  y_list=[]
  for i in df['연도']:
    y_list.append(i)

  for i in range(1,n+1):
    i+=2022
    y_list.append(i)

  future=pd.DataFrame({
      'ds':y_list
  })
  year_future=pd.to_datetime(future['ds'],format='%Y')
  future['ds']=year_future

  date=pd.to_datetime(df['연도'],format='%Y')
  df['연도']=date

  # Prophet 모듈 적용을 위해 rename
  df.rename(columns={'연도':'ds','총 계':'y'},inplace=True)
  
  # df와 future 데이터 프레임 리턴
  return(df,future)
df,future_df=gen_timeseries_df(5)

# Propeht 모듈 적용 : 시계열 예측
# 기존 데이터가 연도를 기준으로 있기 때문에 yearly_seasonality만 적용
m = Prophet(yearly_seasonality=True)
m.fit(df)

#predict: 신뢰구간을 포함한 예측 실행
forecast = m.predict(future_df)

# Plotly를 이용해 시각화
def gen_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'],y=df['y'],mode='markers+lines'))
    fig.add_trace(go.Scatter(x=forecast['ds'],y=forecast['yhat'],mode='markers+lines'))
    return fig

st.plotly_chart(gen_graph())