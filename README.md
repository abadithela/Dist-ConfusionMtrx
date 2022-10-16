## Evaluation Metrics for Object Detection
This repository presents the results corresponding to the ICRA 2023 Submission on Evaluation Metrics for Object Detection for Autonomous Systems by Apurva Badithela, Tichakorn Wongpiromsarn, and Richard M. Murray. The paper presents two types of evaluation metrics -- class-labeled and proposition-labeled  -- -- distance parametrized confusion matrices. A variation of each of these metrics that accounts for distance from ego is also accounted for.

### Class-labeled distance parameterized confusion matrix
Confusion matrix for distance $0 < d <= 10.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &                31 &               0 &            0 \\
 ('obstacle',)   &                 0 &             191 &            0 \\
 ('empty',)      &               127 &             734 &         3227 \\
\hline
\end{tabular}
```
Confusion matrix for distance $10 < d <= 20.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &               158 &               2 &            0 \\
 ('obstacle',)   &                 1 &             998 &            0 \\
 ('empty',)      &               873 &            3526 &         1969 \\
\hline
\end{tabular}
 ```
 
Confusion matrix for distance $20.0 < d <= 30.0$
 ```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &               291 &               4 &            0 \\
 ('obstacle',)   &                 0 &            1052 &            0 \\
 ('empty',)      &              1794 &            4689 &         1387 \\
\hline
\end{tabular}
```

Confusion matrix for distance $30.0 < d <= 40.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &               183 &               5 &            0 \\
 ('obstacle',)   &                 0 &             983 &            0 \\
 ('empty',)      &              1913 &            4132 &         1474 \\
\hline
\end{tabular}
 ```

Confusion matrix for distance $40.0 <= d <= 50.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &                30 &               3 &            0 \\
 ('obstacle',)   &                 0 &             759 &            0 \\
 ('empty',)      &              1396 &            3093 &         1733 \\
\hline
\end{tabular}
 ```
 
Printing confusion matrix for distance $50.0 <= d <= 60.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &                10 &               3 &            0 \\
 ('obstacle',)   &                 0 &             409 &            0 \\
 ('empty',)      &               848 &            2151 &         2188 \\
\hline
\end{tabular}
```
 
Confusion matrix for distance $60.0 <= d <= 70.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &                 5 &               3 &            0 \\
 ('obstacle',)   &                 0 &             171 &            0 \\
 ('empty',)      &               555 &            1167 &         2742 \\
\hline
\end{tabular}
 ```
 
Confusion matrix from distance $70.0 <= d <= 80.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &                 2 &               1 &            0 \\
 ('obstacle',)   &                 0 &              85 &            0 \\
 ('empty',)      &               281 &             761 &         3077 \\
\hline
\end{tabular}
 ```
 
Confusion matrix from distance $80.0 <= d <= 90.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &                 1 &               0 &            0 \\
 ('obstacle',)   &                 0 &              36 &            0 \\
 ('empty',)      &                88 &             409 &         3436 \\
\hline
\end{tabular}
 ```
 
Printing confusion matrix from distance $90.0 <= d <= 100.0$
```math
\begin{tabular}{lrrr}
\hline
                 &   ('pedestrian',) &   ('obstacle',) &   ('empty',) \\
\hline
 ('pedestrian',) &                 0 &               0 &            0 \\
 ('obstacle',)   &                 0 &              12 &            0 \\
 ('empty',)      &                35 &             207 &         3651 \\
\hline
\end{tabular}
```
