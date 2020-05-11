"""

Del 1: Test portion : 2014 - 2018 (5år)

Lagre konfig her så det blir lett å hente ut. S

1. Ingen andre variabler enn skyer i tidligere tidsteg.
    * skriv kode fro denne
2. Predikere skyer basert på trykk, temperatur og to fuktigheter
3. Predikere skyer basert på trykk, temperatur og to fuktigheter + 1 tidssteg bak i tid
4. Predikere skyer basert på trykk, temperatur og to fuktigheter + 2 tidssteg bak i tid
5. Predikere skyer basert på trykk, temperatur og to fuktigheter + 3 tidssteg bak i tid


Kryssvalidering.

Test1 : 2004 - 2008 (5år)
Test2 : 2009 - 2013 (5år)
Test3 : 2014 - 2018 (5år)


"""

test_portions =  [('2004-04-01', '2008-12-31'),
                  ('2009-01-01', '2013-12-31'),
                  ('2014-01-01', '2018-12-31')]:


from AR_model import AR_model
from traditional_AR_model import TRADITIONAL_AR_model

# Test modell
m = AR_model(start = None,      stop = None,
             test_start = '2014-01-01', test_stop = '2018-12-31',
             order = 1,                 transform = False,
             sigmoid = False)

coeff = m.fit()
m.save()
print(m.get_configuration())


# Modell 1: Ikke implementert.
m = TRADITIONAL_AR_model(start = '2012-01-01',      stop = '2012-01-03',
             test_start = '2014-01-01', test_stop = '2018-12-31',
                     order = 1,                 transform = False,
                     sigmoid = False)

coeff = m.fit()
m.save()
print(m.get_configuration())


# Modell 2:
m = AR_model(start = None,      stop = None,
             test_start = '2014-01-01', test_stop = '2018-12-31',
             order = 0,                 transform = True,
             sigmoid = True)
coeff = m.fit()
m.save()
print(m.get_configuration())

# Modell 3:
m = AR_model(start = None,      stop = None,
             test_start = '2014-01-01', test_stop = '2018-12-31',
             order = 1,                 transform = False,
             sigmoid = True)
coeff = m.fit()
m.save()
print(m.get_configuration())


# Modell 4:
m = AR_model(start = None,      stop = None,
             test_start = '2014-01-01', test_stop = '2018-12-31',
             order = 2,                 transform = False,
             sigmoid = True)
coeff = m.fit()
m.save()
print(m.get_configuration())


# Modell 5:
m = AR_model(start = None,      stop = None,
             test_start = '2014-01-01', test_stop = '2018-12-31',
             order = 3,                 transform = False,
             sigmoid = True)
coeff = m.fit()
m.save()
print(m.get_configuration())
