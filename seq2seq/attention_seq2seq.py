# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.
import tensorflow as tf
import numpy as np

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in '%SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑책상배우다고양가르치인간적']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑'],
            ['desk', '책상'], ['cat', '고양이'],
            ['learn', '배우다'], ['teach', '가르치다'],
            ['personal', '인간적인'] ]


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    len_i = [len(seq[0]) for seq in seq_data]
    len_o = [len(seq[1]) for seq in seq_data]

    max_len_i = max(len_i)
    max_len_o = max(len_o)

    for seq in seq_data:
        i = seq[0]
        o = seq[1]
        while not len(i) == max_len_i:
            if len(i) < max_len_i:
                i += '%'
        while not len(o) == max_len_o:
            if len(o) < max_len_o:
                o += '%'

    
        input = [num_dic[n] for n in i]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [num_dic[n] for n in ('S' + o)]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (o + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch
#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 1500
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len


#########
# 신경망 모델 구성
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])


with tf.variable_scope('encode'):
    enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [512, 256, 128]]
    enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
    outputs, enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=enc_input, dtype=tf.float32)

    print(outputs)

with tf.variable_scope('decode'):
    dec_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [512, 256, 128]]
    dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cell)
    outputs, dec_states = tf.nn.dynamic_rnn(cell=dec_cell, inputs=dec_input, initial_state=enc_states,
                                            dtype=tf.float32)
    print(outputs)
'''
model = tf.layers.dense(outputs, n_class, activation=None)


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


#########
# 번역 테스트
######
# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    seq_data = [word, 'P' * len(word)]

    input_batch, output_batch, target_batch = make_batch([seq_data])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    #print(result)

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]

    decoded = ''.join(decoded)
    decoded = decoded.replace('%', '')
    decoded = decoded.replace('E', '')

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    
    #if 'E' in decoded:
    #    end = decoded.index('E')
    #    translated = ''.join(decoded[:end])
    
    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    
    #if '%' in translated:
    #    end = translated.index('%')
    #    translated = ''.join(translated[:end])

    return decoded


print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wood ->', translate('wood'))
print('love ->', translate('love'))
print('learn ->', translate('learn'))
print('teach ->', translate('teach'))
print('personal ->', translate('personal'))
'''