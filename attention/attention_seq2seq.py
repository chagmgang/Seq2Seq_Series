# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.
import tensorflow as tf
import numpy as np
from seq2seq_model import s2s

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
        output = [num_dic[n] for n in ('S' + o)]
        target = [num_dic[n] for n in (o + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch, max_len_i + 1, max_len_o + 1, dic_len

learning_rate = 0.01
n_hidden = 128
total_epoch = 500
n_class = n_input = dic_len

input_batch, output_batch, target_batch, enc_sent_size, output_sent_size, vocab_size = make_batch(seq_data)
S2S = s2s(enc_sent_size, output_sent_size, vocab_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(total_epoch):
    _, loss = sess.run([S2S.optimizer, S2S.cost],
                       feed_dict={S2S.enc_input: input_batch,
                                  S2S.dec_input: output_batch,
                                  S2S.targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']

    while not len(word) == enc_sent_size - 1:
        if len(word) < enc_sent_size - 1:
            word += '%'

    seq_data = [word, 'P' * (output_sent_size - 1)]

    input_batch, output_batch, target_batch, _, _, _ = make_batch([seq_data])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.

    result = sess.run(S2S.prediction,
                      feed_dict={S2S.enc_input: input_batch,
                                 S2S.dec_input: output_batch,
                                 S2S.targets: target_batch})
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
