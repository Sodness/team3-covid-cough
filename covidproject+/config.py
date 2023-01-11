# config.py
db={
'user'     : 'mydata',		# 1)
'password' : 'ghktkwhgdkgkqslek522',		# 2)
'host'     : 'localhost',	# 3)
'port'     : 3306,			# 4)
'database' : 'covid'		# 5)
}

SQLALCHEMY_DATABASE_URI = f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"
SQLALCHEMY_MODIFICATIONS = False
SECRET_KEY = "dev"