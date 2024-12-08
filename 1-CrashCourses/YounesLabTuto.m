%% Section 1
clc; clear; % clear cli + workspace

a = 0 % def a
b = 'Hey there!'; % def b
disp(a)
disp(b)

%% Section 2 
% Sections divide the code in multiple blocks

% Multiple lines
% comment

d = 'Section 2';

d


%% Creating Matrices
clc; clear;

A = [1,2,3,4]; % Vector = 1D Matrix
disp(A)

A * 2
A - A


B = [1,2,3,4; 5,6,7,8; 9,10,11,12]; % Matrix
B = [
    1, 2, 3, 4; 
    5, 6, 7, 8; 
    9, 10, 11, 12
    ];

disp(B)

5 * B
B/5

% B^2 % error: B * B results in incorrect dimensions
B.^2 % B. = pour chaque élément dans B, le mettre à la puissance 2


C = rand(3) % random 3x3 Matrix

inv(C)

C^2

eig(C) % ?


%% If Statements
clc; clear;

% `&` AND gate ; `|` OR gate

x = 1;
y = 1;

if x > 0
    disp('x greater than 0')
end

%if x > 0
%if x > 0 & y > 0
if x > 0 | y > 0
    disp('x or y greater than 0')
else
    disp('x, y equal or lower than 0')
end

%% Loop statements
clc; clear;

vector = [77,27,34,41];

for i = vector
    disp(i)
end

for i = 1:length(vector)
    disp(i)
end

for i = 1:2:8 % pour i allant de 1 à 8 avec un pas de 2
    disp(i)
end

% <<!>> EN MATLAB ON COMMENCE A COMPTER A PARTIR DE 1 !!!
vector(1) % accéder au 1er élément
vector(2)


n = length(vector)
vector(n)


k = 5;
while k > 0
    disp(k)
    k = k-1;
end