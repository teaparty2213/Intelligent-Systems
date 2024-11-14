% 知能システム論 第7回 課題1 一階述語論理を用いたサザエさん一家の知識グラフ
% 藤井智哉(理学部生物情報科学科, 学籍番号: 05235509)

partner(Namihei, Fune).
child(Namihei, Sazae).
child(Namihei, Katsuo).
child(Namihei, Wakame).
chile(Fune, Sazae).
child(Fune, Katsuo).
child(Fune, Wakame).
pet(Namihei, Tama).
partner(Sazae, Masuo).
child(Sazae, Tarao).
friend(Tarao, Rika).
friend(Katuo, Hanazawa).
friend(Katuo, Nakajima).
friend(Hanazawa, Nakajima).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
% あとはクラスの関係と性質を記述すればok