%% 1. Basic arithmetics
clc; clear;

% Par défaut, le résultat va dans la variable `ans`
1+2
1-2
1*2
1/2

(5-8)*3
5-8*3

1 + 2; % hide result ending command by ;

%% 2. Variables
clc; clear;

x = 7;
y = 10;

x+y
x-y
x*y
x/y


temp_1 = 20; % temperature : meaningful names for variables
acc_balance = 3000;


%% 3. Change format
clc; clear; format short;

1/3

format short
1/3

format long
1/3

help format

%% 4. Remove variables
clc; clear;

clear all % not recommanded to only clear workspace variables, use `clear` instead

%% 5. Clear specific variables

%acc_ % press tab to autocomplete

x = 78;
y = 3;
clear x % clear a specific variable

z1 = 88;
z2 = 'Hey';
u = 4;
clear z* % clear all the variables matching (here, all the variables starting by z)

who %indicates current variables
whos  %indicates informations about current vars

a = 1+2; b = 3*4; c = 10+10;

%% 6. Pre-defined constants
clc; clear;

pi
Inf % infinity
NaN
i % imaginary unit


%% 7. Relational Operators
clc; clear;

4 > 5 % 0 = false

5 > 3 % 1 = true

4 >= 4


%% 8. Built-in functions
clc; clear;

sin(5)
cos(0)
sind(20)
exp(4) % exponentiel
log(10)
abs(-2) % absolute value


%% 9. Vectors and Matrices
clc; clear;

A = [1 2 3; 4 5 6; 7 8 9]

whos

b = [1 2 3]'
c = b' % `'` permet de faire une transposée


%% 10. Indexing
clc; clear;

%<<!>> Matlab starts counting from 1 !!!
A = [1 2 3; 4 5 6; 7 8 9]

v = A(:,2) % `:` all the rows ; second column

A(end, :) = 0 % dernière colonne, tous les éléments prennent la valeur 0

[1:5; 6:10; 11:15; 16:20;]

ones(1, size(A,1)) % matrice remplie de 1, de même nbr de colonnes que A, 1 ligne
ones(size(A, 2), 1)

size(A,1)

zeros(5) % 5x5 matrice remplie de zéros
zeros(5,1)
zeros(1,5)

eye(3) % Matrice identité de taille 3x3

A
A(1,2) % 1re ligne, 2ème colonne
A(2,2) = 10


%% 11. Other keywords
clc; clear;
A = [1 2 3; 4 5 6; 7 8 9]


length(A)
size(A)
numel(A) % nbr elts in A
trace(A) % somme des éléments sur la diagonale (1+5+9)


%% 12. 3 commons matrix operations
clc; clear;
A = [1 2 3; 4 10 6; 0 0 0]

eig(A)
inv(A)


A = [1 2 3; 4 5 6; 7 8 9]
inv(A)

det(A) % déterminant de la matrice


%% 13. Matrix operations
clc; clear;
A = rand(5)
B = rand(5)

A + B
A - B
A * B % multiplication de 2 matrices
A / B
A ^ 2 % A * A

A .* B % multiplication élément par élément entre les 2 matrices !!!!
A .^ 2 % tous les éléments de la matrice A au carré

A > B % matrice contenant des true/false


%% 14. Solve System of Linear Equations
clc; clear;

A = [1 2 3; 4 5 6; 7 8 9]
b = [ 1 1 1]

x = A\b' 
% Résoudre le système :
% x + 2y + 3z = 1
% 4x + 5y + 6z = 1
% 7x + 8y + 9z = 1


%% 15. M-file scripts

%% 16. 3 Magic C's
clear;
close all % close all opened figures
clc;


%% 17. Loops

counter = 0;

for i = 1:5
    counter = counter + 1;
    disp(counter)
end

counter = 10;

while counter > 5
    counter = counter - 1;
    disp(counter)
end

%% 18. Plotting
clc; close; clear;

x = 0:0.1:5
y = x.^2

plot(x, y, 'r+')
title("My first plot!!!!")
xlabel('x-value')
ylabel('y-value')
grid on
hold

y2 = x.^3;
y3 = x.^4;
plot(x, y2, 'g*')
plot(x,y3)
hold off
legend('Plot1', 'Plot2', 'Plot3')


%% 19. Subplotting
subplot(311)
plot(x,y)
subplot(312)
plot(x,y2)
subplot(313)
plot(x,y3)


%% 20. Functions
clc; clear;
help triangle_area

triangle_area(5,10)


%% 21. Debugging code



