import os
import sys

REPO_URL = "https://github.com/Akseleu-J/Atomic_AI_hybrid-v0.1.git"
REPO_DIR = "Atomic_AI_hybrid-v0.1"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL}
else:
    !cd {REPO_DIR} && git pull 

if REPO_DIR not in sys.path:
    sys.path.append(os.path.abspath(REPO_DIR))

print("✅ Репозиторий готов, пути добавлены.")
Cloning into 'Atomic_AI_hybrid-v0.1'...
remote: Enumerating objects: 229, done.
remote: Counting objects:   1% (1/78)
remote: Counting objects:   2% (2/78)
remote: Counting objects:   3% (3/78)
remote: Counting objects:   5% (4/78)
remote: Counting objects:   6% (5/78)
remote: Counting objects:   7% (6/78)
remote: Counting objects:   8% (7/78)
remote: Counting objects:  10% (8/78)
remote: Counting objects:  11% (9/78)
remote: Counting objects:  12% (10/78)
remote: Counting objects:  14% (11/78)
remote: Counting objects:  15% (12/78)
remote: Counting objects:  16% (13/78)
remote: Counting objects:  17% (14/78)
remote: Counting objects:  19% (15/78)
remote: Counting objects:  20% (16/78)
remote: Counting objects:  21% (17/78)
remote: Counting objects:  23% (18/78)
remote: Counting objects:  24% (19/78)
remote: Counting objects:  25% (20/78)
remote: Counting objects:  26% (21/78)
remote: Counting objects:  28% (22/78)
remote: Counting objects:  29% (23/78)
remote: Counting objects:  30% (24/78)
remote: Counting objects:  32% (25/78)
remote: Counting objects:  33% (26/78)
remote: Counting objects:  34% (27/78)
remote: Counting objects:  35% (28/78)
remote: Counting objects:  37% (29/78)
remote: Counting objects:  38% (30/78)
remote: Counting objects:  39% (31/78)
remote: Counting objects:  41% (32/78)
remote: Counting objects:  42% (33/78)
remote: Counting objects:  43% (34/78)
remote: Counting objects:  44% (35/78)
remote: Counting objects:  46% (36/78)
remote: Counting objects:  47% (37/78)
remote: Counting objects:  48% (38/78)
remote: Counting objects:  50% (39/78)
remote: Counting objects:  51% (40/78)
remote: Counting objects:  52% (41/78)
remote: Counting objects:  53% (42/78)
remote: Counting objects:  55% (43/78)
remote: Counting objects:  56% (44/78)
remote: Counting objects:  57% (45/78)
remote: Counting objects:  58% (46/78)
remote: Counting objects:  60% (47/78)
remote: Counting objects:  61% (48/78)
remote: Counting objects:  62% (49/78)
remote: Counting objects:  64% (50/78)
remote: Counting objects:  65% (51/78)
remote: Counting objects:  66% (52/78)
remote: Counting objects:  67% (53/78)
remote: Counting objects:  69% (54/78)
remote: Counting objects:  70% (55/78)
remote: Counting objects:  71% (56/78)
remote: Counting objects:  73% (57/78)
remote: Counting objects:  74% (58/78)
remote: Counting objects:  75% (59/78)
remote: Counting objects:  76% (60/78)
remote: Counting objects:  78% (61/78)
remote: Counting objects:  79% (62/78)
remote: Counting objects:  80% (63/78)
remote: Counting objects:  82% (64/78)
remote: Counting objects:  83% (65/78)
remote: Counting objects:  84% (66/78)
remote: Counting objects:  85% (67/78)
remote: Counting objects:  87% (68/78)
remote: Counting objects:  88% (69/78)
remote: Counting objects:  89% (70/78)
remote: Counting objects:  91% (71/78)
remote: Counting objects:  92% (72/78)
remote: Counting objects:  93% (73/78)
remote: Counting objects:  94% (74/78)
remote: Counting objects:  96% (75/78)
remote: Counting objects:  97% (76/78)
remote: Counting objects:  98% (77/78)
remote: Counting objects: 100% (78/78)
remote: Counting objects: 100% (78/78), done.
remote: Compressing objects:   1% (1/78)
remote: Compressing objects:   2% (2/78)
remote: Compressing objects:   3% (3/78)
remote: Compressing objects:   5% (4/78)
remote: Compressing objects:   6% (5/78)
remote: Compressing objects:   7% (6/78)
remote: Compressing objects:   8% (7/78)
remote: Compressing objects:  10% (8/78)
remote: Compressing objects:  11% (9/78)
remote: Compressing objects:  12% (10/78)
remote: Compressing objects:  14% (11/78)
remote: Compressing objects:  15% (12/78)
remote: Compressing objects:  16% (13/78)
remote: Compressing objects:  17% (14/78)
remote: Compressing objects:  19% (15/78)
remote: Compressing objects:  20% (16/78)
remote: Compressing objects:  21% (17/78)
remote: Compressing objects:  23% (18/78)
remote: Compressing objects:  24% (19/78)
remote: Compressing objects:  25% (20/78)
remote: Compressing objects:  26% (21/78)
remote: Compressing objects:  28% (22/78)
remote: Compressing objects:  29% (23/78)
remote: Compressing objects:  30% (24/78)
remote: Compressing objects:  32% (25/78)
remote: Compressing objects:  33% (26/78)
remote: Compressing objects:  34% (27/78)
remote: Compressing objects:  35% (28/78)
remote: Compressing objects:  37% (29/78)
remote: Compressing objects:  38% (30/78)
remote: Compressing objects:  39% (31/78)
remote: Compressing objects:  41% (32/78)
remote: Compressing objects:  42% (33/78)
remote: Compressing objects:  43% (34/78)
remote: Compressing objects:  44% (35/78)
remote: Compressing objects:  46% (36/78)
remote: Compressing objects:  47% (37/78)
remote: Compressing objects:  48% (38/78)
remote: Compressing objects:  50% (39/78)
remote: Compressing objects:  51% (40/78)
remote: Compressing objects:  52% (41/78)
remote: Compressing objects:  53% (42/78)
remote: Compressing objects:  55% (43/78)
remote: Compressing objects:  56% (44/78)
remote: Compressing objects:  57% (45/78)
remote: Compressing objects:  58% (46/78)
remote: Compressing objects:  60% (47/78)
remote: Compressing objects:  61% (48/78)
remote: Compressing objects:  62% (49/78)
remote: Compressing objects:  64% (50/78)
remote: Compressing objects:  65% (51/78)
remote: Compressing objects:  66% (52/78)
remote: Compressing objects:  67% (53/78)
remote: Compressing objects:  69% (54/78)
remote: Compressing objects:  70% (55/78)
remote: Compressing objects:  71% (56/78)
remote: Compressing objects:  73% (57/78)
remote: Compressing objects:  74% (58/78)
remote: Compressing objects:  75% (59/78)
remote: Compressing objects:  76% (60/78)
remote: Compressing objects:  78% (61/78)
remote: Compressing objects:  79% (62/78)
remote: Compressing objects:  80% (63/78)
remote: Compressing objects:  82% (64/78)
remote: Compressing objects:  83% (65/78)
remote: Compressing objects:  84% (66/78)
remote: Compressing objects:  85% (67/78)
remote: Compressing objects:  87% (68/78)
remote: Compressing objects:  88% (69/78)
remote: Compressing objects:  89% (70/78)
remote: Compressing objects:  91% (71/78)
remote: Compressing objects:  92% (72/78)
remote: Compressing objects:  93% (73/78)
remote: Compressing objects:  94% (74/78)
remote: Compressing objects:  96% (75/78)
remote: Compressing objects:  97% (76/78)
remote: Compressing objects:  98% (77/78)
remote: Compressing objects: 100% (78/78)
remote: Compressing objects: 100% (78/78), done.
Receiving objects:   0% (1/229)
Receiving objects:   1% (3/229)
Receiving objects:   2% (5/229)
Receiving objects:   3% (7/229)
Receiving objects:   4% (10/229)
Receiving objects:   5% (12/229)
Receiving objects:   6% (14/229)
Receiving objects:   7% (17/229)
Receiving objects:   8% (19/229)
Receiving objects:   9% (21/229)
Receiving objects:  10% (23/229)
Receiving objects:  11% (26/229)
Receiving objects:  12% (28/229)
Receiving objects:  13% (30/229)
Receiving objects:  14% (33/229)
Receiving objects:  15% (35/229)
Receiving objects:  16% (37/229)
Receiving objects:  17% (39/229)
Receiving objects:  18% (42/229)
Receiving objects:  19% (44/229)
Receiving objects:  20% (46/229)
Receiving objects:  21% (49/229)
Receiving objects:  22% (51/229)
Receiving objects:  23% (53/229)
Receiving objects:  24% (55/229)
Receiving objects:  25% (58/229)
Receiving objects:  26% (60/229)
Receiving objects:  27% (62/229)
Receiving objects:  28% (65/229)
Receiving objects:  29% (67/229)
Receiving objects:  30% (69/229)
Receiving objects:  31% (71/229)
Receiving objects:  32% (74/229)
Receiving objects:  33% (76/229)
Receiving objects:  34% (78/229)
Receiving objects:  35% (81/229)
Receiving objects:  36% (83/229)
Receiving objects:  37% (85/229)
Receiving objects:  38% (88/229)
Receiving objects:  39% (90/229)
Receiving objects:  40% (92/229)
Receiving objects:  41% (94/229)
Receiving objects:  42% (97/229)
Receiving objects:  43% (99/229)
Receiving objects:  44% (101/229)
Receiving objects:  45% (104/229)
Receiving objects:  46% (106/229)
Receiving objects:  47% (108/229)
Receiving objects:  48% (110/229)
Receiving objects:  49% (113/229)
Receiving objects:  50% (115/229)
Receiving objects:  51% (117/229)
Receiving objects:  52% (120/229)
Receiving objects:  53% (122/229)
Receiving objects:  54% (124/229)
Receiving objects:  55% (126/229)
Receiving objects:  56% (129/229)
Receiving objects:  57% (131/229)
Receiving objects:  58% (133/229)
Receiving objects:  59% (136/229)
Receiving objects:  60% (138/229)
remote: Total 229 (delta 46), reused 0 (delta 0), pack-reused 151 (from 1)
Receiving objects:  61% (140/229)
Receiving objects:  62% (142/229)
Receiving objects:  63% (145/229)
Receiving objects:  64% (147/229)
Receiving objects:  65% (149/229)
Receiving objects:  66% (152/229)
Receiving objects:  67% (154/229)
Receiving objects:  68% (156/229)
Receiving objects:  69% (159/229)
Receiving objects:  70% (161/229)
Receiving objects:  71% (163/229)
Receiving objects:  72% (165/229)
Receiving objects:  73% (168/229)
Receiving objects:  74% (170/229)
Receiving objects:  75% (172/229)
Receiving objects:  76% (175/229)
Receiving objects:  77% (177/229)
Receiving objects:  78% (179/229)
Receiving objects:  79% (181/229)
Receiving objects:  80% (184/229)
Receiving objects:  81% (186/229)
Receiving objects:  82% (188/229)
Receiving objects:  83% (191/229)
Receiving objects:  84% (193/229)
Receiving objects:  85% (195/229)
Receiving objects:  86% (197/229)
Receiving objects:  87% (200/229)
Receiving objects:  88% (202/229)
Receiving objects:  89% (204/229)
Receiving objects:  90% (207/229)
Receiving objects:  91% (209/229)
Receiving objects:  92% (211/229)
Receiving objects:  93% (213/229)
Receiving objects:  94% (216/229)
Receiving objects:  95% (218/229)
Receiving objects:  96% (220/229)
Receiving objects:  97% (223/229)
Receiving objects:  98% (225/229)
Receiving objects:  99% (227/229)
Receiving objects: 100% (229/229)
Receiving objects: 100% (229/229), 131.07 KiB | 2.52 MiB/s, done.
Resolving deltas:   0% (0/135)
Resolving deltas:   1% (2/135)
Resolving deltas:   2% (3/135)
Resolving deltas:   5% (7/135)
Resolving deltas:   6% (9/135)
Resolving deltas:   7% (10/135)
Resolving deltas:  10% (14/135)
Resolving deltas:  11% (15/135)
Resolving deltas:  12% (17/135)
Resolving deltas:  14% (20/135)
Resolving deltas:  15% (21/135)
Resolving deltas:  16% (22/135)
Resolving deltas:  17% (23/135)
Resolving deltas:  19% (26/135)
Resolving deltas:  20% (27/135)
Resolving deltas:  22% (31/135)
Resolving deltas:  23% (32/135)
Resolving deltas:  24% (33/135)
Resolving deltas:  25% (34/135)
Resolving deltas:  27% (37/135)
Resolving deltas:  28% (38/135)
Resolving deltas:  30% (41/135)
Resolving deltas:  31% (43/135)
Resolving deltas:  32% (44/135)
Resolving deltas:  35% (48/135)
Resolving deltas:  36% (49/135)
Resolving deltas:  37% (51/135)
Resolving deltas:  38% (52/135)
Resolving deltas:  39% (53/135)
Resolving deltas:  40% (54/135)
Resolving deltas:  41% (56/135)
Resolving deltas:  42% (57/135)
Resolving deltas:  43% (59/135)
Resolving deltas:  44% (60/135)
Resolving deltas:  45% (61/135)
Resolving deltas:  46% (63/135)
Resolving deltas:  47% (64/135)
Resolving deltas:  48% (65/135)
Resolving deltas:  49% (67/135)
Resolving deltas:  50% (68/135)
Resolving deltas:  51% (69/135)
Resolving deltas:  52% (71/135)
Resolving deltas:  54% (73/135)
Resolving deltas:  55% (75/135)
Resolving deltas:  56% (76/135)
Resolving deltas:  58% (79/135)
Resolving deltas:  61% (83/135)
Resolving deltas:  62% (84/135)
Resolving deltas:  63% (86/135)
Resolving deltas:  64% (87/135)
Resolving deltas:  65% (88/135)
Resolving deltas:  66% (90/135)
Resolving deltas:  67% (91/135)
Resolving deltas:  68% (92/135)
Resolving deltas:  73% (99/135)
Resolving deltas:  74% (100/135)
Resolving deltas:  75% (102/135)
Resolving deltas:  76% (103/135)
Resolving deltas:  77% (104/135)
Resolving deltas:  78% (106/135)
Resolving deltas:  79% (107/135)
Resolving deltas:  80% (108/135)
Resolving deltas:  81% (110/135)
Resolving deltas:  82% (111/135)
Resolving deltas:  83% (113/135)
Resolving deltas:  84% (114/135)
Resolving deltas:  85% (116/135)
Resolving deltas:  86% (117/135)
Resolving deltas:  87% (118/135)
Resolving deltas:  88% (119/135)
Resolving deltas:  89% (121/135)
Resolving deltas:  90% (122/135)
Resolving deltas:  92% (125/135)
Resolving deltas:  93% (126/135)
Resolving deltas:  94% (127/135)
Resolving deltas:  95% (129/135)
Resolving deltas:  97% (131/135)
Resolving deltas:  98% (133/135)
Resolving deltas:  99% (134/135)
Resolving deltas: 100% (135/135)
Resolving deltas: 100% (135/135), done.
✅ Репозиторий готов, пути добавлены.
!rm -rf Atomic_AI_hybrid-v0.1
!git clone https://github.com/Akseleu-J/Atomic_AI_hybrid-v0.1.git
Cloning into 'Atomic_AI_hybrid-v0.1'...
remote: Enumerating objects: 229, done.
remote: Counting objects:   1% (1/78)
remote: Counting objects:   2% (2/78)
remote: Counting objects:   3% (3/78)
remote: Counting objects:   5% (4/78)
remote: Counting objects:   6% (5/78)
remote: Counting objects:   7% (6/78)
remote: Counting objects:   8% (7/78)
remote: Counting objects:  10% (8/78)
remote: Counting objects:  11% (9/78)
remote: Counting objects:  12% (10/78)
remote: Counting objects:  14% (11/78)
remote: Counting objects:  15% (12/78)
remote: Counting objects:  16% (13/78)
remote: Counting objects:  17% (14/78)
remote: Counting objects:  19% (15/78)
remote: Counting objects:  20% (16/78)
remote: Counting objects:  21% (17/78)
remote: Counting objects:  23% (18/78)
remote: Counting objects:  24% (19/78)
remote: Counting objects:  25% (20/78)
remote: Counting objects:  26% (21/78)
remote: Counting objects:  28% (22/78)
remote: Counting objects:  29% (23/78)
remote: Counting objects:  30% (24/78)
remote: Counting objects:  32% (25/78)
remote: Counting objects:  33% (26/78)
remote: Counting objects:  34% (27/78)
remote: Counting objects:  35% (28/78)
remote: Counting objects:  37% (29/78)
remote: Counting objects:  38% (30/78)
remote: Counting objects:  39% (31/78)
remote: Counting objects:  41% (32/78)
remote: Counting objects:  42% (33/78)
remote: Counting objects:  43% (34/78)
remote: Counting objects:  44% (35/78)
remote: Counting objects:  46% (36/78)
remote: Counting objects:  47% (37/78)
remote: Counting objects:  48% (38/78)
remote: Counting objects:  50% (39/78)
remote: Counting objects:  51% (40/78)
remote: Counting objects:  52% (41/78)
remote: Counting objects:  53% (42/78)
remote: Counting objects:  55% (43/78)
remote: Counting objects:  56% (44/78)
remote: Counting objects:  57% (45/78)
remote: Counting objects:  58% (46/78)
remote: Counting objects:  60% (47/78)
remote: Counting objects:  61% (48/78)
remote: Counting objects:  62% (49/78)
remote: Counting objects:  64% (50/78)
remote: Counting objects:  65% (51/78)
remote: Counting objects:  66% (52/78)
remote: Counting objects:  67% (53/78)
remote: Counting objects:  69% (54/78)
remote: Counting objects:  70% (55/78)
remote: Counting objects:  71% (56/78)
remote: Counting objects:  73% (57/78)
remote: Counting objects:  74% (58/78)
remote: Counting objects:  75% (59/78)
remote: Counting objects:  76% (60/78)
remote: Counting objects:  78% (61/78)
remote: Counting objects:  79% (62/78)
remote: Counting objects:  80% (63/78)
remote: Counting objects:  82% (64/78)
remote: Counting objects:  83% (65/78)
remote: Counting objects:  84% (66/78)
remote: Counting objects:  85% (67/78)
remote: Counting objects:  87% (68/78)
remote: Counting objects:  88% (69/78)
remote: Counting objects:  89% (70/78)
remote: Counting objects:  91% (71/78)
remote: Counting objects:  92% (72/78)
remote: Counting objects:  93% (73/78)
remote: Counting objects:  94% (74/78)
remote: Counting objects:  96% (75/78)
remote: Counting objects:  97% (76/78)
remote: Counting objects:  98% (77/78)
remote: Counting objects: 100% (78/78)
remote: Counting objects: 100% (78/78), done.
remote: Compressing objects:   1% (1/78)
remote: Compressing objects:   2% (2/78)
remote: Compressing objects:   3% (3/78)
remote: Compressing objects:   5% (4/78)
remote: Compressing objects:   6% (5/78)
remote: Compressing objects:   7% (6/78)
remote: Compressing objects:   8% (7/78)
remote: Compressing objects:  10% (8/78)
remote: Compressing objects:  11% (9/78)
remote: Compressing objects:  12% (10/78)
remote: Compressing objects:  14% (11/78)
remote: Compressing objects:  15% (12/78)
remote: Compressing objects:  16% (13/78)
remote: Compressing objects:  17% (14/78)
remote: Compressing objects:  19% (15/78)
remote: Compressing objects:  20% (16/78)
remote: Compressing objects:  21% (17/78)
remote: Compressing objects:  23% (18/78)
remote: Compressing objects:  24% (19/78)
remote: Compressing objects:  25% (20/78)
remote: Compressing objects:  26% (21/78)
remote: Compressing objects:  28% (22/78)
remote: Compressing objects:  29% (23/78)
remote: Compressing objects:  30% (24/78)
remote: Compressing objects:  32% (25/78)
remote: Compressing objects:  33% (26/78)
remote: Compressing objects:  34% (27/78)
remote: Compressing objects:  35% (28/78)
remote: Compressing objects:  37% (29/78)
remote: Compressing objects:  38% (30/78)
remote: Compressing objects:  39% (31/78)
remote: Compressing objects:  41% (32/78)
remote: Compressing objects:  42% (33/78)
remote: Compressing objects:  43% (34/78)
remote: Compressing objects:  44% (35/78)
remote: Compressing objects:  46% (36/78)
remote: Compressing objects:  47% (37/78)
remote: Compressing objects:  48% (38/78)
remote: Compressing objects:  50% (39/78)
remote: Compressing objects:  51% (40/78)
remote: Compressing objects:  52% (41/78)
remote: Compressing objects:  53% (42/78)
remote: Compressing objects:  55% (43/78)
remote: Compressing objects:  56% (44/78)
remote: Compressing objects:  57% (45/78)
remote: Compressing objects:  58% (46/78)
remote: Compressing objects:  60% (47/78)
remote: Compressing objects:  61% (48/78)
remote: Compressing objects:  62% (49/78)
remote: Compressing objects:  64% (50/78)
remote: Compressing objects:  65% (51/78)
remote: Compressing objects:  66% (52/78)
remote: Compressing objects:  67% (53/78)
remote: Compressing objects:  69% (54/78)
remote: Compressing objects:  70% (55/78)
remote: Compressing objects:  71% (56/78)
remote: Compressing objects:  73% (57/78)
remote: Compressing objects:  74% (58/78)
remote: Compressing objects:  75% (59/78)
remote: Compressing objects:  76% (60/78)
remote: Compressing objects:  78% (61/78)
remote: Compressing objects:  79% (62/78)
remote: Compressing objects:  80% (63/78)
remote: Compressing objects:  82% (64/78)
remote: Compressing objects:  83% (65/78)
remote: Compressing objects:  84% (66/78)
remote: Compressing objects:  85% (67/78)
remote: Compressing objects:  87% (68/78)
remote: Compressing objects:  88% (69/78)
remote: Compressing objects:  89% (70/78)
remote: Compressing objects:  91% (71/78)
remote: Compressing objects:  92% (72/78)
remote: Compressing objects:  93% (73/78)
remote: Compressing objects:  94% (74/78)
remote: Compressing objects:  96% (75/78)
remote: Compressing objects:  97% (76/78)
remote: Compressing objects:  98% (77/78)
remote: Compressing objects: 100% (78/78)
remote: Compressing objects: 100% (78/78), done.
Receiving objects:   0% (1/229)
Receiving objects:   1% (3/229)
Receiving objects:   2% (5/229)
Receiving objects:   3% (7/229)
Receiving objects:   4% (10/229)
Receiving objects:   5% (12/229)
Receiving objects:   6% (14/229)
Receiving objects:   7% (17/229)
Receiving objects:   8% (19/229)
Receiving objects:   9% (21/229)
Receiving objects:  10% (23/229)
Receiving objects:  11% (26/229)
Receiving objects:  12% (28/229)
Receiving objects:  13% (30/229)
Receiving objects:  14% (33/229)
Receiving objects:  15% (35/229)
Receiving objects:  16% (37/229)
Receiving objects:  17% (39/229)
Receiving objects:  18% (42/229)
Receiving objects:  19% (44/229)
Receiving objects:  20% (46/229)
Receiving objects:  21% (49/229)
Receiving objects:  22% (51/229)
Receiving objects:  23% (53/229)
Receiving objects:  24% (55/229)
Receiving objects:  25% (58/229)
Receiving objects:  26% (60/229)
Receiving objects:  27% (62/229)
Receiving objects:  28% (65/229)
Receiving objects:  29% (67/229)
Receiving objects:  30% (69/229)
Receiving objects:  31% (71/229)
Receiving objects:  32% (74/229)
Receiving objects:  33% (76/229)
Receiving objects:  34% (78/229)
Receiving objects:  35% (81/229)
Receiving objects:  36% (83/229)
Receiving objects:  37% (85/229)
Receiving objects:  38% (88/229)
Receiving objects:  39% (90/229)
Receiving objects:  40% (92/229)
Receiving objects:  41% (94/229)
Receiving objects:  42% (97/229)
Receiving objects:  43% (99/229)
Receiving objects:  44% (101/229)
Receiving objects:  45% (104/229)
Receiving objects:  46% (106/229)
Receiving objects:  47% (108/229)
Receiving objects:  48% (110/229)
Receiving objects:  49% (113/229)
Receiving objects:  50% (115/229)
Receiving objects:  51% (117/229)
Receiving objects:  52% (120/229)
Receiving objects:  53% (122/229)
Receiving objects:  54% (124/229)
Receiving objects:  55% (126/229)
Receiving objects:  56% (129/229)
Receiving objects:  57% (131/229)
Receiving objects:  58% (133/229)
Receiving objects:  59% (136/229)
Receiving objects:  60% (138/229)
Receiving objects:  61% (140/229)
remote: Total 229 (delta 46), reused 0 (delta 0), pack-reused 151 (from 1)
Receiving objects:  62% (142/229)
Receiving objects:  63% (145/229)
Receiving objects:  64% (147/229)
Receiving objects:  65% (149/229)
Receiving objects:  66% (152/229)
Receiving objects:  67% (154/229)
Receiving objects:  68% (156/229)
Receiving objects:  69% (159/229)
Receiving objects:  70% (161/229)
Receiving objects:  71% (163/229)
Receiving objects:  72% (165/229)
Receiving objects:  73% (168/229)
Receiving objects:  74% (170/229)
Receiving objects:  75% (172/229)
Receiving objects:  76% (175/229)
Receiving objects:  77% (177/229)
Receiving objects:  78% (179/229)
Receiving objects:  79% (181/229)
Receiving objects:  80% (184/229)
Receiving objects:  81% (186/229)
Receiving objects:  82% (188/229)
Receiving objects:  83% (191/229)
Receiving objects:  84% (193/229)
Receiving objects:  85% (195/229)
Receiving objects:  86% (197/229)
Receiving objects:  87% (200/229)
Receiving objects:  88% (202/229)
Receiving objects:  89% (204/229)
Receiving objects:  90% (207/229)
Receiving objects:  91% (209/229)
Receiving objects:  92% (211/229)
Receiving objects:  93% (213/229)
Receiving objects:  94% (216/229)
Receiving objects:  95% (218/229)
Receiving objects:  96% (220/229)
Receiving objects:  97% (223/229)
Receiving objects:  98% (225/229)
Receiving objects:  99% (227/229)
Receiving objects: 100% (229/229)
Receiving objects: 100% (229/229), 131.07 KiB | 2.43 MiB/s, done.
Resolving deltas:   0% (0/135)
Resolving deltas:   1% (2/135)
Resolving deltas:   2% (3/135)
Resolving deltas:   5% (7/135)
Resolving deltas:   7% (10/135)
Resolving deltas:   8% (11/135)
Resolving deltas:   9% (13/135)
Resolving deltas:  12% (17/135)
Resolving deltas:  13% (18/135)
Resolving deltas:  14% (19/135)
Resolving deltas:  15% (21/135)
Resolving deltas:  17% (23/135)
Resolving deltas:  19% (26/135)
Resolving deltas:  20% (27/135)
Resolving deltas:  21% (29/135)
Resolving deltas:  22% (30/135)
Resolving deltas:  23% (32/135)
Resolving deltas:  24% (33/135)
Resolving deltas:  25% (34/135)
Resolving deltas:  26% (36/135)
Resolving deltas:  28% (38/135)
Resolving deltas:  31% (42/135)
Resolving deltas:  34% (47/135)
Resolving deltas:  35% (48/135)
Resolving deltas:  36% (49/135)
Resolving deltas:  37% (51/135)
Resolving deltas:  38% (52/135)
Resolving deltas:  39% (53/135)
Resolving deltas:  40% (55/135)
Resolving deltas:  41% (56/135)
Resolving deltas:  42% (58/135)
Resolving deltas:  43% (59/135)
Resolving deltas:  45% (61/135)
Resolving deltas:  49% (67/135)
Resolving deltas:  50% (68/135)
Resolving deltas:  51% (69/135)
Resolving deltas:  53% (72/135)
Resolving deltas:  54% (73/135)
Resolving deltas:  55% (75/135)
Resolving deltas:  57% (77/135)
Resolving deltas:  58% (79/135)
Resolving deltas:  59% (80/135)
Resolving deltas:  60% (81/135)
Resolving deltas:  61% (83/135)
Resolving deltas:  62% (84/135)
Resolving deltas:  63% (86/135)
Resolving deltas:  68% (93/135)
Resolving deltas:  69% (94/135)
Resolving deltas:  71% (96/135)
Resolving deltas:  72% (98/135)
Resolving deltas:  73% (99/135)
Resolving deltas:  74% (100/135)
Resolving deltas:  75% (102/135)
Resolving deltas:  76% (103/135)
Resolving deltas:  77% (104/135)
Resolving deltas:  78% (106/135)
Resolving deltas:  79% (107/135)
Resolving deltas:  80% (108/135)
Resolving deltas:  81% (110/135)
Resolving deltas:  82% (112/135)
Resolving deltas:  85% (115/135)
Resolving deltas:  86% (117/135)
Resolving deltas:  88% (119/135)
Resolving deltas:  89% (121/135)
Resolving deltas:  90% (122/135)
Resolving deltas:  91% (123/135)
Resolving deltas:  92% (125/135)
Resolving deltas:  93% (126/135)
Resolving deltas:  94% (127/135)
Resolving deltas:  95% (129/135)
Resolving deltas:  96% (130/135)
Resolving deltas:  97% (131/135)
Resolving deltas:  98% (133/135)
Resolving deltas:  99% (134/135)
Resolving deltas: 100% (135/135)
Resolving deltas: 100% (135/135), done.
pip install -U "jax[tpu]"
Requirement already satisfied: jax[tpu] in /usr/local/lib/python3.12/site-packages (0.10.2)
Collecting jax[tpu]
  Downloading jax-0.11.0-py3-none-any.whl.metadata (13 kB)
Collecting jaxlib<=0.11.0,>=0.11.0 (from jax[tpu])
  Downloading jaxlib-0.11.0-cp312-cp312-manylinux_2_27_x86_64.whl.metadata (1.3 kB)
Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (0.5.4)
Requirement already satisfied: numpy>=2.1 in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (2.5.0)
Requirement already satisfied: opt_einsum in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (3.4.0)
Requirement already satisfied: scipy>=1.15 in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (1.18.0)
Collecting libtpu==0.0.44.* (from jax[tpu])
  Downloading libtpu-0.0.44.1-cp312-cp312-manylinux_2_31_x86_64.whl.metadata (1.5 kB)
Requirement already satisfied: requests in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (2.34.2)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (3.4.7)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (3.18)
Requirement already satisfied: urllib3<3,>=1.26 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (2.7.0)
Requirement already satisfied: certifi>=2023.5.7 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (2026.6.17)
Downloading libtpu-0.0.44.1-cp312-cp312-manylinux_2_31_x86_64.whl (215.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/215.6 MB ? eta -:--:--
   ━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.5/215.6 MB 222.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━ 87.0/215.6 MB 234.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━ 130.8/215.6 MB 228.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━ 175.4/215.6 MB 226.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 215.5/215.6 MB 225.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 215.6/215.6 MB 75.4 MB/s eta 0:00:00
Downloading jaxlib-0.11.0-cp312-cp312-manylinux_2_27_x86_64.whl (87.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/87.3 MB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━ 43.8/87.3 MB 220.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 87.0/87.3 MB 223.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 87.0/87.3 MB 223.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 87.0/87.3 MB 223.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 87.0/87.3 MB 223.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.3/87.3 MB 77.4 MB/s eta 0:00:00
Downloading jax-0.11.0-py3-none-any.whl (3.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/3.3 MB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 74.5 MB/s eta 0:00:00
Installing collected packages: libtpu, jaxlib, jax
  Attempting uninstall: libtpu
    Found existing installation: libtpu 0.0.17
    Uninstalling libtpu-0.0.17:
      Successfully uninstalled libtpu-0.0.17
  Attempting uninstall: jaxlib
    Found existing installation: jaxlib 0.10.2
    Uninstalling jaxlib-0.10.2:
      Successfully uninstalled jaxlib-0.10.2
  Attempting uninstall: jax
    Found existing installation: jax 0.10.2
    Uninstalling jax-0.10.2:
      Successfully uninstalled jax-0.10.2
Successfully installed jax-0.11.0 jaxlib-0.11.0 libtpu-0.0.44.1
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
[notice] A new release of pip is available: 25.0.1 -> 26.2.1
[notice] To update, run: pip install --upgrade pip
Note: you may need to restart the kernel to use updated packages.
import glob
import os
import re
import time
import signal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from model import FullHybridMoEModel, ModelConfig, set_model_mesh, get_model_mesh
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str


def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int, seq_len: int = 8192):
    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]

    if batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by n_devices={n_devices}."
        )

    batch_axis = "tpu_nodes"
    set_model_mesh(mesh, batch_axis=batch_axis)

    tx = make_hybrid_optimizer(total_steps=total_steps)
    model = FullHybridMoEModel(cfg=config)

    init_rng = jax.random.PRNGKey(0)
    abstract_params = jax.eval_shape(
        lambda: model.init(init_rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))
    )["params"]

    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))

    MIN_SHARD_SIZE = 128

    def _get_shard_spec(path, param):
        if not hasattr(param, "shape") or param.ndim == 0:
            return NamedSharding(mesh, P())
        if "experts_block" in path_to_str(path):
            return NamedSharding(mesh, P(*([None] * param.ndim)))
        best_axis, best_size = None, -1
        for i, size in enumerate(param.shape):
            if size % n_devices == 0 and (size // n_devices) >= MIN_SHARD_SIZE and size > best_size:
                best_axis, best_size = i, size
        if best_axis is None:
            return NamedSharding(mesh, P(*([None] * param.ndim)))
        spec = [None] * param.ndim
        spec[best_axis] = "tpu_nodes"
        return NamedSharding(mesh, P(*spec))

    param_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, abstract_params)

    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))
    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, opt_state_abstract)

    def model_apply_wrapped(variables, input_ids, rngs=None, deterministic=True, **kwargs):
        return model.apply(
            variables, input_ids,
            rngs=rngs, deterministic=deterministic,
            **kwargs
        )

    def distributed_train_step(p, s, b, r):
        loss_fn = lambda param: compute_loss(
            param, model_apply_wrapped, b, config,
            rngs={"dropout": r},
            deterministic=False, return_aux=True,
            ce_chunk_size=2048
        )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, new_s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), new_s, loss, aux_info

    def distributed_val_step(p, b):
        return compute_loss(
            p, model_apply_wrapped, b, config,
            rngs=None,
            deterministic=True
        )

    aux_info_sharding = {
        "ce_loss": NamedSharding(mesh, P()),
        "aux_loss": NamedSharding(mesh, P()),
        "z_loss": NamedSharding(mesh, P()),
        "expert_utilization": NamedSharding(mesh, P(None, None)),
    }
    compiled_train = jax.jit(
        distributed_train_step,
        donate_argnums=(0, 1),
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            {"input_ids": data_sharding, "labels": data_sharding},
            NamedSharding(mesh, P(None)),
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            NamedSharding(mesh, P()),
            aux_info_sharding,
        ),
    )
    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )
    return compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding


def resolve_source_files(output_dir, prefix):
    merged_ids = os.path.join(output_dir, f"{prefix}_input_ids.npy")
    merged_lbls = os.path.join(output_dir, f"{prefix}_labels.npy")
    if os.path.exists(merged_ids) and os.path.exists(merged_lbls):
        return [(merged_ids, merged_lbls)]

    shard_ids_paths = sorted(
        glob.glob(os.path.join(output_dir, f"{prefix}_shard_ids_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    pairs = []
    for ids_path in shard_ids_paths:
        lbls_path = ids_path.replace("_shard_ids_", "_shard_lbls_")
        if os.path.exists(lbls_path):
            pairs.append((ids_path, lbls_path))
    if not pairs:
        raise FileNotFoundError(
            f"Не найдены файлы для prefix={prefix!r} в {output_dir} -- ни объединённого "
            f"{prefix}_input_ids.npy, ни шардов {prefix}_shard_ids_*.npy. Проверьте путь."
        )
    return pairs


def build_manifest(file_pairs):
    manifest = []
    total = 0
    for ids_path, lbls_path in file_pairs:
        n_rows = np.load(ids_path, mmap_mode="r").shape[0]
        manifest.append((ids_path, lbls_path, n_rows))
        total += n_rows
        print(f"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков")
    print(f"[DATA] Комбинированный пул: {total:,} блоков из {len(manifest)} файл(ов)")
    return manifest


def dataloader_multi_source(file_pairs, batch_size, data_sharding, seq_len, val_split=0.05):
    manifest = build_manifest(file_pairs)
    sizes = np.array([n for _, _, n in manifest])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    total_blocks = int(offsets[-1])
    context_length = np.load(manifest[0][0], mmap_mode="r").shape[1]
    if context_length > seq_len:
        context_length = seq_len

    mmap_cache = {}

    def _get_mmap(path):
        arr = mmap_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            mmap_cache[path] = arr
        return arr

    def _gather_batch(global_indices):
        shard_of = np.searchsorted(offsets, global_indices, side="right") - 1
        ids_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        lbls_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        for s in np.unique(shard_of):
            m = shard_of == s
            local_idx = global_indices[m] - offsets[s]
            ids_path, lbls_path, _ = manifest[s]
            ids_full = _get_mmap(ids_path)[local_idx]
            lbls_full = _get_mmap(lbls_path)[local_idx]
            ids_out[m] = ids_full[:, :seq_len]
            lbls_out[m] = lbls_full[:, :seq_len]
        return ids_out, lbls_out

    val_size = int(total_blocks * val_split)
    train_size = total_blocks - val_size

    all_idx = np.arange(total_blocks)
    np.random.RandomState(42).shuffle(all_idx)
    train_idx_pool = all_idx[:train_size]
    val_idx_pool = all_idx[train_size:]

    def _generator(pool, is_train=True):
        idx_local = np.copy(pool)
        local_rng = np.random.RandomState(123)
        while True:
            if is_train:
                local_rng.shuffle(idx_local)
            for step in range(len(idx_local) // batch_size):
                batch_idx = idx_local[step * batch_size: (step + 1) * batch_size]
                ids_np, lbls_np = _gather_batch(batch_idx)
                yield {
                    "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                    "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                }
            if not is_train:
                break

    return (
        _generator(train_idx_pool, True),
        lambda: _generator(val_idx_pool, False),
        train_size // batch_size,
        val_size // batch_size,
    )


def main_execution():
    config = ModelConfig(
        d_model=512,
        d_state=128,
        d_conv=4,
        expand=2,
        n_heads=8,
        d_latent=256,
        d_ff=3072,
        num_experts=8,
        top_k=2,
        num_layers=22,
        vocab_size=151936,
        dropout_rate=0.1,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.0001,
        moe_capacity_factor=1.25,
        tie_embeddings=True,
        label_smoothing=0.05,
        router_noise_std=0.3,
        use_flash_attention=True,
        deltanet_chunk_size=256
    )

    file_pairs = [
    (
        "/kaggle/input/datasets/akseleu1j/atentic-data/agentic_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/atentic-data/agentic_labels.npy",
    ),
    (
        "/kaggle/input/datasets/akseleu1j/coding-labels/coding_A_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/coding-labels/coding_A_labels.npy",
    ),
    (
        "/kaggle/input/datasets/akseleu1j/coding-ids/coding_B_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/coding-ids/coding_B_labels.npy",
    ),
    (
        "/kaggle/input/datasets/akseleu1j/simple-data/common_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/simple-data/common_labels.npy",
    ),
    (
        "/kaggle/input/datasets/akseleu1j/math-ids/math_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/math-ids/math_labels.npy",
    ),
    (
        "/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_input_ids.npy",
        "/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_labels.npy",
    ),
    ]

    for ids_path, lbls_path in file_pairs:
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Не найден файл: {ids_path}")
        if not os.path.exists(lbls_path):
            raise FileNotFoundError(f"Не найден файл: {lbls_path}")
    print("Все файлы найдены.")

    manifest = build_manifest(file_pairs)
    total_blocks = sum(n for _, _, n in manifest)
    print(f"Всего блоков: {total_blocks:,}")

    batch_size = 8
    seq_len = 4096
    epochs = 1
    early_stop_patience = 2
    eval_every_steps = 1000
    eval_batches = 40
    eval_patience = 4

    val_split = 0.05
    val_size = int(total_blocks * val_split)
    train_size = total_blocks - val_size
    train_steps_per_epoch = train_size // batch_size
    total_train_steps = train_steps_per_epoch * epochs

    print(f"[TPU] Компиляция XLA графа под {total_train_steps} общих шагов обучения "
          f"({epochs} эпох(и) x {train_steps_per_epoch} шагов)...")
    compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding = (
        make_shard_and_compile(config, total_train_steps, batch_size, seq_len)
    )
    print(f"[TPU] Устройств в mesh: {mesh.shape['tpu_nodes']} (FSDP: params, state и батч шардированы).")

    train_stream, val_factory, train_steps, val_steps = dataloader_multi_source(
        file_pairs, batch_size, data_sharding, seq_len=seq_len
    )

    global_rng = jax.random.PRNGKey(42)
    init_params_fn = jax.jit(
        lambda rng: model.init(rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))["params"],
        out_shardings=param_sharding,
    )
    params = init_params_fn(global_rng)
    print(f"[MEM] Доступно памяти на чипе 0: {jax.local_devices()[0].memory_stats()}")
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Общее количество параметров: {total_params:,} (≈ {total_params / 1e9:.2f} млрд)")

    weights_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))
    n_devices_display = mesh.shape["tpu_nodes"]
    print(f"Размер весов модели (глобально): {weights_bytes / 1e9:.2f} ГБ "
          f"(с FSDP на чип реально хранится в среднем ~{weights_bytes / 1e9 / n_devices_display:.2f} ГБ -- "
          "точная цифра зависит от того, какие оси делимы на n_devices, см. _get_shard_spec)")

    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    _dummy_batch = {
        "input_ids": jax.device_put(jnp.zeros((batch_size, seq_len), dtype=jnp.int32), data_sharding),
        "labels": jax.device_put(jnp.zeros((batch_size, seq_len), dtype=jnp.int32), data_sharding),
    }
    _lowered = compiled_train.lower(params, opt_state, _dummy_batch, global_rng)
    _compiled_exec = _lowered.compile()
    _analysis = _compiled_exec.memory_analysis()
    print(f"[MEM ANALYSIS] HBM temp:      {_analysis.temp_size_in_bytes / 1e9:.2f} ГБ")
    print(f"[MEM ANALYSIS] HBM arguments: {_analysis.argument_size_in_bytes / 1e9:.2f} ГБ")
    print(f"[MEM ANALYSIS] HBM output:    {_analysis.output_size_in_bytes / 1e9:.2f} ГБ")
    print("[TPU] Компиляция готова -- переходим к реальному обучению.")

    checkpoint_dir = "/kaggle/working/orbax_checkpoints"
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer(), options)
    best_checkpoint_dir = "/kaggle/working/orbax_checkpoints_best"
    best_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    best_mngr = ocp.CheckpointManager(best_checkpoint_dir, ocp.StandardCheckpointer(), best_options)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    best_eval_loss = float("inf")
    eval_no_improve_count = 0
    stopped_early = False

    total_tokens_processed = 0
    epoch_start_time = time.perf_counter()

    for epoch in range(epochs):
        for step in range(train_steps):
            global_rng, step_rng = jax.random.split(global_rng)

            _t0 = time.perf_counter()
            batch = next(train_stream)
            _t_data = time.perf_counter() - _t0

            total_tokens_processed += batch_size * seq_len

            # ПРОФИЛИРОВКА (шаги 5-9, после разогрева/компиляции): jax.profiler
            # пишет реальный op-level trace -- какие именно HLO-операции (MoE
            # argsort/scatter/gather, Muon-матмулы, GDN/Mamba associative_scan,
            # обычные Dense-матмулы) сколько времени реально едят на устройстве.
            # Открывается в TensorBoard (`pip install tensorboard-plugin-profile`,
            # затем `%load_ext tensorboard` + `%tensorboard --logdir /kaggle/working/jax-trace`
            # в отдельной ячейке) либо через https://ui.perfetto.dev (загрузить файл
            # из /kaggle/working/jax-trace напрямую, plugin не обязателен).
            _do_trace = 5 <= step < 10
            if _do_trace and step == 5:
                jax.profiler.start_trace("/kaggle/working/jax-trace")

            _t1 = time.perf_counter()
            params, opt_state, train_loss, aux_info = compiled_train(params, opt_state, batch, step_rng)
            if step < 30:
                jax.block_until_ready(train_loss)
            _t_compute = time.perf_counter() - _t1

            if _do_trace and step == 9:
                jax.block_until_ready(train_loss)
                jax.profiler.stop_trace()
                print("[PROFILER] Трейс шагов 5-9 сохранён в /kaggle/working/jax-trace")

            if step < 30:
                print(f"[TIMING] step {step}: данные={_t_data*1000:.0f}мс  "
                      f"TPU={_t_compute*1000:.0f}мс  "
                      f"(доля данных: {_t_data/(_t_data+_t_compute)*100:.0f}%)")

            global_step += 1

            if step % 10 == 0:
                print(
                    f"Epoch: {epoch} | Step: {step}/{train_steps} | "
                    f"Global Step: {global_step} | Train Loss: {jax.device_get(train_loss):.4f} "
                    f"(ce={jax.device_get(aux_info['ce_loss']):.4f} "
                    f"aux={jax.device_get(aux_info['aux_loss']):.4f} "
                    f"z={jax.device_get(aux_info['z_loss']):.5f})"
                )
                if aux_info["expert_utilization"] is not None:
                    util = jax.device_get(aux_info["expert_utilization"])
                    util_std_per_layer = util.std(axis=-1)
                    worst_layer = int(util_std_per_layer.argmax())
                    print(
                        f"           expert utilization std (max over layers, layer {worst_layer}): "
                        f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts}"
                    )
            if (step+1) % 10 == 0:
                print(f"[Успех] Тестовой запуск успешно проверен!")
                os.kill(os.getpid(), signal.SIGKILL)
            if global_step % eval_every_steps == 0:
                val_stream = val_factory()
                eval_loss = 0.0
                n_batches_done = 0
                for _ in range(eval_batches):
                    try:
                        eval_batch = next(val_stream)
                    except StopIteration:
                        break
                    eval_loss += jax.device_get(compiled_val(params, eval_batch))
                    n_batches_done += 1
                eval_loss /= max(n_batches_done, 1)
                print(f"[EVAL] Step {global_step}: val loss (частичный, {n_batches_done} батчей) = {eval_loss:.4f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    eval_no_improve_count = 0
                else:
                    eval_no_improve_count += 1
                    if eval_no_improve_count >= eval_patience:
                        print(
                            f"[EARLY STOP] Частичный val loss не улучшался {eval_patience} "
                            "проверок подряд. Останавливаю обучение немедленно."
                        )
                        mngr.save(global_step, args=ocp.args.StandardSave(params))
                        best_mngr.save(global_step, args=ocp.args.StandardSave(params))
                        print(f"[ORBAX] Финальный чекпоинт (шаг {global_step}) сохранён в оба каталога.")
                        stopped_early = True
                        break

        if stopped_early:
            break

        print(f"--- Эпоха {epoch} завершена. Запуск распределенной кросс-валидации ---")
        val_stream = val_factory()
        total_val_loss = 0.0
        for _ in range(val_steps):
            total_val_loss += jax.device_get(compiled_val(params, next(val_stream)))

        mean_val_loss = total_val_loss / val_steps
        print(f"===> Эпоха: {epoch} | ИТОГОВЫЙ СРЕДНИЙ VALIDATION LOSS: {mean_val_loss:.4f} <===")

        epoch_elapsed = time.perf_counter() - epoch_start_time
        tokens_per_sec = total_tokens_processed / epoch_elapsed
        print(f"Средняя скорость эпохи: {tokens_per_sec / 1e6:.2f} млн токенов/сек")

        total_tokens_processed = 0
        epoch_start_time = time.perf_counter()

        mngr.save(global_step, args=ocp.args.StandardSave(params))
        print(f"[ORBAX] Чекпоинт для шага {global_step} успешно зафиксирован.")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0
            best_mngr.save(global_step, args=ocp.args.StandardSave(params))
            print(f"[ORBAX] Новый лучший val loss ({best_val_loss:.4f}) -- сохранён в {best_checkpoint_dir}")
        else:
            epochs_without_improvement += 1
            print(
                f"[EARLY STOP] val loss не улучшился {epochs_without_improvement} эпох(и) подряд "
                f"(лучший: {best_val_loss:.4f})"
            )
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"[EARLY STOP] Останавливаю обучение -- val loss не улучшался "
                    f"{early_stop_patience} эпохи подряд. Лучшие веса лежат в {best_checkpoint_dir}."
                )
                break

    print("Обучение завершено.")


if __name__ == "__main__":
    main_execution()
/usr/local/lib/python3.12/site-packages/jax/_src/cloud_tpu_init.py:88: UserWarning: Transparent hugepages are not enabled. TPU runtime startup and shutdown time should be significantly improved on TPU v5e and newer. If not already set, you may need to enable transparent hugepages in your VM image (sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled")
  warnings.warn(
Все файлы найдены.
[DATA] agentic_input_ids.npy: 146,695 блоков
[DATA] coding_A_input_ids.npy: 310,058 блоков
[DATA] coding_B_input_ids.npy: 544,433 блоков
[DATA] common_input_ids.npy: 274,658 блоков
[DATA] math_input_ids.npy: 127,656 блоков
[DATA] reasoning_input_ids.npy: 457,763 блоков
[DATA] Комбинированный пул: 1,861,263 блоков из 6 файл(ов)
Всего блоков: 1,861,263
[TPU] Компиляция XLA графа под 221025 общих шагов обучения (1 эпох(и) x 221025 шагов)...
WARNING: Logging before InitGoogle() is written to STDERR
E0000 00:00:1785940598.615268      73 common_lib.cc:943] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8471 in any of the 0 ports provided in `tpu_process_addresses`="local"
=== Source Location Trace: ===
learning/45eac/tfrc/runtime/common_lib.cc:239
[TPU] Устройств в mesh: 8 (FSDP: params, state и батч шардированы).
[DATA] agentic_input_ids.npy: 146,695 блоков
[DATA] coding_A_input_ids.npy: 310,058 блоков
[DATA] coding_B_input_ids.npy: 544,433 блоков
[DATA] common_input_ids.npy: 274,658 блоков
[DATA] math_input_ids.npy: 127,656 блоков
[DATA] reasoning_input_ids.npy: 457,763 блоков
[DATA] Комбинированный пул: 1,861,263 блоков из 6 файл(ов)
[MEM] Доступно памяти на чипе 0: {'num_allocs': 1044, 'bytes_in_use': 2353927680, 'peak_bytes_in_use': 2353927680, 'largest_alloc_size': 44040192, 'bytes_limit': 16909332480, 'bytes_reserved': 66560, 'peak_bytes_reserved': 66560, 'bytes_reservable_limit': 14573867008, 'largest_free_block_bytes': 14555338240}
Общее количество параметров: 768,072,434 (≈ 0.77 млрд)
Размер весов модели (глобально): 3.07 ГБ (с FSDP на чип реально хранится в среднем ~0.38 ГБ -- точная цифра зависит от того, какие оси делимы на n_devices, см. _get_shard_spec)
WARNING:absl:Configured `CheckpointManager` using deprecated legacy API. Please follow the instructions at https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html to migrate.
WARNING:absl:Configured `CheckpointManager` using deprecated legacy API. Please follow the instructions at https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html to migrate.
[MEM ANALYSIS] HBM temp:      4.63 ГБ
[MEM ANALYSIS] HBM arguments: 2.45 ГБ
[MEM ANALYSIS] HBM output:    2.45 ГБ
[TPU] Компиляция готова -- переходим к реальному обучению.
[TIMING] step 0: данные=270мс  TPU=7252мс  (доля данных: 4%)
Epoch: 0 | Step: 0/221025 | Global Step: 1 | Train Loss: 12.3129 (ce=12.3129 aux=0.0000 z=0.00000)
[TIMING] step 1: данные=109мс  TPU=6865мс  (доля данных: 2%)
[TIMING] step 2: данные=77мс  TPU=6864мс  (доля данных: 1%)
[TIMING] step 3: данные=81мс  TPU=6864мс  (доля данных: 1%)
[TIMING] step 4: данные=83мс  TPU=6865мс  (доля данных: 1%)
[TIMING] step 5: данные=77мс  TPU=6865мс  (доля данных: 1%)
[TIMING] step 6: данные=77мс  TPU=6866мс  (доля данных: 1%)
[TIMING] step 7: данные=77мс  TPU=6864мс  (доля данных: 1%)
[TIMING] step 8: данные=75мс  TPU=6865мс  (доля данных: 1%)
import os
import glob

def find_dataset_pairs(
    base_path="/kaggle/input",
    prefixes=None,  # если None, то автоматически определяются
):
    """
    Находит все пары input_ids / labels .npy файлов в /kaggle/input.
    Возвращает список кортежей (ids_path, lbls_path).
    """
    if prefixes is None:
        # Автоматически определяем префиксы по именам файлов
        ids_files = glob.glob(os.path.join(base_path, "**", "*_input_ids.npy"), recursive=True)
        prefixes = sorted({os.path.basename(f).replace("_input_ids.npy", "") for f in ids_files})
    
    pairs = []
    for prefix in prefixes:
        # Ищем файлы по шаблону
        pattern = os.path.join(base_path, "**", f"{prefix}_input_ids.npy")
        ids_files = glob.glob(pattern, recursive=True)
        for ids_path in ids_files:
            lbls_path = ids_path.replace("_input_ids.npy", "_labels.npy")
            if os.path.exists(lbls_path):
                pairs.append((ids_path, lbls_path))
    return pairs

# Находим все пары
file_pairs = find_dataset_pairs()

# Выводим в формате, удобном для копирования в train.py
print("file_pairs = [")
for ids_path, lbls_path in file_pairs:
    print(f"    (\n        \"{ids_path}\",\n        \"{lbls_path}\",\n    ),")
print("]")
    file_pairs = [
        (
            "/kaggle/input/datasets/akseleu1j/agentic-datasetids-and-labels/processed_jax_data/agentic_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/agentic-datasetids-and-labels/processed_jax_data/agentic_labels.npy",
        ),#0.81B tokens
        (
            "/kaggle/input/datasets/akseleu1j/coding-ids/coding_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/coding-labels/coding_labels.npy",
        ),#2.5B tokens
        (
            "/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/reasoning-labels/reasoning_labels.npy",
        ),#2.5B tokens
        (
            "/kaggle/input/datasets/akseleu1j/math-ids/new_data_ids.npy",
            "/kaggle/input/datasets/akseleu1j/math-labels/new_data_labels.npy",
        ),#2.6B tokens
    ]
pip install tensorboard-plugin-profile
%load_ext tensorboard + %tensorboard --logdir /kaggle/working/jax-trace
 
