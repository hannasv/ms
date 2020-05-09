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





from sclouds.ml.regrression.AR_model import AR_model

m = AR_model(start = '2012-01-01',      stop = '2012-01-03',
             test_start = '2012-03-01', test_stop = '2012-03-03',
             order = 1,                 transform = True,
             sigmoid = False)
coeff = m.fit()
m.save()
