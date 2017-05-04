"""
Temel kullanım:

Tensorflow C++ da yazıdı bu yüzden pythondaki verileri doğrudan kütüphanede kullanamıyoruz.
Verileri kullanabilmek için tensorflowun bize sunduğu veri yapılarını kullanmamız gerekiyor.

Variable() değişkenleri, constant() sabit değerleri, placeholder() geçici olarak değer tutan veri yapılarını temsil ediyor.
Bu veri yapılarında çeşitli boyutlardaki matrisleri tutabiliriz. 
Veri türü olarak python listelerini, numpy dizilerini veya tensorflow dizilerini kullanabiliriz.
Ayrıca istersek tanımladığımız veri yapılarını adlandırabiliriz. 

Kullanımı:
    v = tf.Variable(matris, name="isim")
    p = tf.placeholder(matris, name="isim")
    c = tf.constant(matris, name="isim")

Tensorflow öntanımlı olarak 32 bit float veriler ile işlemler yapıyor.
Eğer Variable() tanımlamışsak tensorflowun  bu değişkenleri tanıması için 
tf.global_variables_initializer() fonksiyonunu çalıştırmamız gerekiyor.

Yapay Sinir Hucresi:
    Sinyal = Agirlik_Matrisi x Girdi + Sapma Degeri
    aktivasyon_fonksiyonu(Sinyal)
seklinde tanımlanıyor

Agirlik matrisleri (weights) yapay sinir aglarindaki hafizayi temsil ediyor
Yapay Sinir Aginda bu matrisleri, gelen veriler ve sonrasinda kullandigimiz optimizasyon algoritmasi
ile sekillendiriyoruz. Yapay sinir agi boyle ogreniyor.

Agirlik matrisini sadece girdilerle sekillendirerek aktivasyon fonksiyonunu esnek bir sekilde kullanmayiz. 
Aciklamak gerekirse y = ax dogrusu orijinden gecer ve a parametresiyle sadece dogrunun egimini degistirebiliriz.
Eger bu denkleme b adinda bir degisken daha eklersek dogruyu y ekseninde kaydirabiliriz. 
Bu yuzden Sapma Degeri (Bias) ekliyoruz. S = Wx + B (y = ax + b)

Aktivasyon fonksiyonu yapay sinir hucresinin urettigi sinyalin gucune gore hucrenin aktif olup olmayacagini
veya ne olcude aktif olacagini belirler.
sigmoid, tanh, step 

Yapay zekanın ogrenmesini saglamak icin tahmin edilen veriler ile dogru verileri kıyaslayıp,
aralarındaki kaybi(hatayi) hesaplayip azaltmamiz gerekiyor.
Bu hatayi hesaplamak icin;
loss = tf.reduce_mean(tf.square(y_pred - y_real))
islemini gerceklestirdik. Burada oncelikle iki deger arasindaki farkı hesapladık. 
Ardindan belirginlestirmek icin farkin karesini aldik. Son olarak da reduce_mean() 
fonksiyonu ile kaybin ortalama degerini olctuk. Bu fonksiyon parametre olarak 
input_tensor (giris matrisini) aliyor. Ek olarak hangi boyuta gore ortalama alinacagi
belirlenebiliyor.

Bu hatayi optimizasyon algoritmalari ile azaltabiliriz. Tensorflow bize hazir kullanabilecegimiz 
bazi optimizasyon algoritmalari sunuyor. Biz bunlar arasindan GradientDescentOptimizer'i kullanacagiz.
Bu algoritma N boyutlu hata matrisinde minumum noktayi bulmayi hedefliyor.
Bu algoritmayi;
optimizer = tf.train.GradientDescentOptimizer(0.5)
seklinde kullanabiliriz. Girdigimiz 0.5 degeri ogrenme oranini temsil ediyor. 
Bu degeri cok yuksek yaparsak minumum noktayi es gecebiliriz, cok kucuk yaparsak da
minimum noktaya ulasmamiz cok uzun surebilir. 

Tensorflow "lazy" calisiyor yani tanimladigimiz degiskenler, fonsiyonlar, optimizasyon algoritmalari 
tanimladigimiz anda calistirilmiyor. Bu islemleri aktif etmek icin tensorflow oturumu (Session) kullanmamiz
gerekiyor. Session'u bir isaretci olarak dusunulebilir. Tanimladigimiz yapay sinir agi modelinin istedigimiz
adimini Session ile calistirip degerini alabiliriz.

Asagida en basit haliyle bir yapay sinir agi olusturduk. Giris verileri icin (X) -1 ile 1 arasinda rastgele sayilardan olusan 3x3 bir matris olusturduk.
Gercek verileri temsil etmek icinde giris verilerinde biraz oynama yaptik. Bu ornekteki amacimiz giris matrisini(X) ile yapay sinir agini egiterek her adimda
gercek degerlere(y_real) biraz daha yaklasmak. 
"""

import numpy as np
import tensorflow as tf

# generate some data
X = np.random.rand(3, 3).astype(np.float32)
y_real = X * 0.1 + 0.3
print(y_real)

# structre start
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 1D array,  [-0.322323, 0.612312, ... , 0.712342]
biases = tf.Variable(tf.zeros([1])) # [0]

y_pred = weights * X + biases # giris verilerine gore bir tahmin olustur (sinir hucresi)

loss = tf.reduce_mean(tf.square(y_pred - y_real)) # kaybi hesapla

optimizer = tf.train.GradientDescentOptimizer(0.5) # learning_rate < 1
train = optimizer.minimize(loss) # modeli optimize et

init = tf.global_variables_initializer() # important, degiskenleri tanimla

# structre end

# Train
sess = tf.Session() # pointer
sess.run(init) # degiskenler tanimlandi
for step in range(1000):
    sess.run(train) # Modeli 1 kez egittik. train -> optimizer -> loss -> (y_pred, y_real) -> Wx_plus_B -> W, X, B

    if step % 200 is 0: # 200 adimda bir
        print(step, sess.run(y_pred), sess.run(biases)) # agirlik matrisini ve sapma degerini yazdirdik. (Amacimiz  y_pred'in, y_real'e yaklasmasi)

