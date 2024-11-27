% 知能システム論 第7回 課題1 一階述語論理を用いたサザエさん一家の知識グラフ
% 藤井智哉(理学部生物情報科学科, 学籍番号: 05235509)

% クラス
subset_of(human, animal).
subset_of(dog, animal).
subset_of(cat, animal).
subset_of(male, human).
subset_of(female, human).

% インスタンス
male(namihei).
male(katsuo).
male(masuo).
male(tarao).
male(nakajima).
female(fune).
female(wakame).
female(sazae).
female(hanazawa).
female(rika).
cat(tama).

% 関係
% 配偶者，妻，夫
partner(fune, namihei).
partner(sazae, masuo).
partner(X, Y) :- partner(Y, X).
wife_of(X, Y) :- partner(X, Y), female(X).
husband_of(X, Y) :- partner(X, Y), male(X).

% ペット
pet_of(tama, namihei).

% 親，子，兄弟，おじ，おば
child_of(katsuo, namihei).
child_of(wakame, namihei).
child_of(sazae, namihei).
child_of(tarao, sazae).
child_of(X, Y) :- child_of(X, Z), partner(Z, Y), Z \= Y.
parent_of(X, Y) :- child_of(Y, X).
sibling_of(X, Y) :- child_of(X, Z), child_of(Y, Z), X \= Y.
uncle_of(X, Y) :- sibling_of(X, Z), parent_of(Z, Y), male(X).
aunt_of(X, Y) :- sibling_of(X, Z), parent_of(Z, Y), female(X).

% 祖先，子孫
ancestor_of(X, Y) :- parent_of(X, Y).
ancestor_of(X, Y) :- parent_of(X, Z), ancestor_of(Z, Y).
descendant_of(X, Y) :- ancestor_of(Y, X).

% 友達
friend(katsuo, hanazawa).
friend(katsuo, nakajima).
friend(hanazawa, nakajima).
friend(tarao, rika).
friend(X, Y) :- friend(Y, X).