#%% Thanks for REF! : https://acdongpgm.tistory.com/174
#%% csv를 db에 넣기 REF : https://velog.io/@99mon/Mysql-CSV-%ED%8C%8C%EC%9D%BC-%EB%84%A3%EA%B8%B0


#%% init-------------------------------------------------------------------
import sqlite3

dbpath = "DB_SQLSTK.db"

conn = sqlite3.connect(dbpath)
cur = conn.cursor()


#%% 데이터 저장하면..---------------------------------------------------------
script = """
DROP TABLE IF EXISTS employees;

CREATE TABLE employees(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT NOT NULL, 
salary REAL,
department TEXT,
position TEXT,
hireDate TEXT);

INSERT INTO employees(name, salary, department, position, hireDate) VALUES('Dave', 300, 'Marketing', 'LV1', '2020-01-01');
INSERT INTO employees(name, salary, department, position, hireDate) VALUES('Clara', 420, 'Sales', 'LV2', '2018-01-11');
INSERT INTO employees(id, name, salary, department, position, hireDate) VALUES(3, 'Jane', 620, 'Developer', 'LV4', '2015-11-01');
INSERT INTO employees VALUES(4, 'Peter', 530, 'Developer', 'LV2', '2020-11-01'); 
"""

cur.executescript(script)
conn.commit() # 실제로 DB에 위 Table & Data가 저장된다.

#%% 데이터 가져오기.. -----------------------------------------------------
'''
fetchall() : 모든 데이터를 한꺼번에 클라이언트로 가져올 때 사용
getchone() : 한번 호출에 하나의 row 만을 가져올 때 사용
fetchmany(n) : n개 만큼의 데이터를 한꺼번에 가져올 때 사용.
'''
cur.execute("SELECT * FROM '240627';")

#%%
employee_list = cur.fetchall()
for employee in employee_list:
    print(employee)
    
    
#%%
import pandas as pd

df = pd.read_sql_query("SELECT * FROM '240627'", conn) 