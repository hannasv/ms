"""

Del 1: Test portion : 2014 - 2018 (5aar)

Lagre konfig her saa det blir lett aa hente ut. S

1. Ingen andre variabler enn skyer i tidligere tidsteg.
    * skriv kode fro denne
2. Predikere skyer basert paa trykk, temperatur og to fuktigheter
3. Predikere skyer basert paa trykk, temperatur og to fuktigheter + 1 tidssteg bak i tid
4. Predikere skyer basert paa trykk, temperatur og to fuktigheter + 2 tidssteg bak i tid
5. Predikere skyer basert paa trykk, temperatur og to fuktigheter + 3 tidssteg bak i tid


Kryssvalidering.

Test1 : 2004 - 2008 (5aar)
Test2 : 2009 - 2013 (5aar)
Test3 : 2014 - 2018 (5aar)


"""

test_portions =  [('2004-04-01', '2008-12-31'),
                  ('2009-01-01', '2013-12-31'),
                  ('2014-01-01', '2018-12-31')]


from AR_model import AR_model
from traditional_AR_model import TRADITIONAL_AR_model


start = '2012-01-01'
stop = '2012-01-03'
test_start = '2014-01-01'
test_stop = '2014-01-03'

transform = True
order = [0, 1, 2, 3, 5]



# Modell 1: Ikke implementert.
m = TRADITIONAL_AR_model(start = start,      stop = stop,
                         test_start = test_start, test_stop = test_stop,
                         order = 1,                 transform = transform,
                         sigmoid = False)

coeff = m.fit()
m.save()
print(m.get_configuration())

print('FINISHED ONE')

# Modell 1: Ikke implementert.
m = TRADITIONAL_AR_model(start = start,      stop = stop,
                         test_start = test_start, test_stop = test_stop,
                         order = 2,                 transform = transform,
                         sigmoid = False)

coeff = m.fit()
m.save()
print(m.get_configuration())

print('FINISHED TWO')



# Test modell
m = AR_model(start = start, stop = stop,
             test_start = test_start, test_stop = test_stop,
             order = 0, transform = transform,
             sigmoid = False)

coeff = m.fit()
m.save()
print(m.get_configuration())

print('FINISHED THREE')

# Test modell
m = AR_model(start = start, stop = stop,
             test_start = test_start, test_stop = test_stop,
             order = 1, transform = transform,
             sigmoid = False)

coeff = m.fit()
m.save()
print(m.get_configuration())

print('FINISHED FOUR')

# Test modell
m = AR_model(start = start, stop = stop,
             test_start = test_start, test_stop = test_stop,
             order = 2, transform = transform,
             sigmoid = False)

coeff = m.fit()
m.save()
print(m.get_configuration())

print('FINISHED FIVE')




# Modell 2:
"""
m = AR_model(start = None,      stop = None,
             test_start = '2014-01-01', test_stop = '2018-12-31',
             order = 0,                 transform = True,
             coeff = m.fit()
             sigmoid = True)
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
"""
