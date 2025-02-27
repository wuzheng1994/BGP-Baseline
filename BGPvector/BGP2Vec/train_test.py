#训练BGP2vec模型，测试用
from bgp2vec import BGP2VEC
bgp_model = BGP2VEC('Models/Word2Vec_No_Indexed_3_iters_5_negative_1_window.word2vec', 'oix-full-snapshot-2018-03-01-0200')
asn_embedding = bgp_model.asn2vec('3356')
print("ASN 3356 embedding:", asn_embedding)