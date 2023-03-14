# push test
## Following Steven Schmatz’s example, I looked at the OEIS entry. A000055 - OEIS

Not everybody’s comfortable with generating functions, but we can perhaps turn it into a recurrence. The entry contains just such a Mathematica formula, credited to Jean-François Alcover:

b[0] = 0;  
b[1] = 1;  
b[n_] := b[n] = Sum[d*b[d]*b[n-j], {j, 1, n-1}, {d, Divisors[j]}]/(n-1); 
a[0] = 1;  
a[n_] := b[n] - (Sum[b[k]*b[n-k], {k, 0, n}] -  
         If[Mod[n, 2] == 0, b[n/2], 0])/2; 
The recursive formula b[n] corresponds to A000081 - OEIS, the number of rooted unlabeled trees. Written in more usual mathematical notation, we have

𝑏(𝑛)=1𝑛−1∑𝑗=1𝑛−1∑𝑑∣𝑗𝑑⋅𝑏(𝑑)⋅𝑏(𝑛−𝑗)
 

𝑎(2𝑛)=𝑏(2𝑛)+12𝑏(𝑛)−12∑𝑘=02𝑛𝑏(𝑘)𝑏(2𝑛−𝑘)
 

𝑎(2𝑛+1)=𝑏(2𝑛+1)−12∑𝑘=02𝑛+1𝑏(𝑘)𝑏(2𝑛+1−𝑘)
 

We can informally justify that by substituting  𝑏(𝑛)
  in for the coefficients of  𝑇(𝑛)
  in the generating function

𝐴(𝑥)=1+𝑇(𝑥)−𝑇2(𝑥)/2+𝑇(𝑥2)/2
 

𝐴(𝑥)=1+∑𝑖𝑏(𝑖)𝑥𝑖−12(∑𝑖𝑏(𝑖)𝑥𝑖)2+12∑𝑖𝑏(2𝑖)𝑥2𝑖
 

Thus, the coefficient on  𝑥𝑛
  will be

𝑏(𝑛)
  (from the first term)
minus half the product of  𝑏(𝑥)
  and  𝑏(𝑦)
  such that  𝑥+𝑦=𝑛
  (from the second term)
plus half of  𝑏(𝑛/2)
 , if  𝑛/2
  is an integer.
So that must be the formula for  𝑎(𝑛)
 . It is unlikely that a nice closed form exists.

Can we explain the recurrence (or the generating function) in terms of the number of rooted trees? Unfortunately, the proof seems nontrivial. (I gave up after noodling with it for a bit.) One can be found in Michael Drmota’s article ”Combinatorics and Asymptotics on Trees” (section 1.3) but involves a bijection created out of a 6-way partition of the rooted trees and a 3-way partition of the unrooted trees.