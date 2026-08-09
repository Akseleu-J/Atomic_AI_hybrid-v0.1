{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d24212b2",
   "metadata": {
    "_cell_guid": "ec7f3703-5d01-42ac-ab91-ccaee3b2b866",
    "_uuid": "3e5a6771-0c4f-4208-842e-7ecff57d49e9",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-08-08T19:37:51.527524Z",
     "iopub.status.busy": "2026-08-08T19:37:51.527332Z",
     "iopub.status.idle": "2026-08-08T19:38:25.413830Z",
     "shell.execute_reply": "2026-08-08T19:38:25.412986Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 33.891279,
     "end_time": "2026-08-08T19:38:25.414926+00:00",
     "exception": false,
     "start_time": "2026-08-08T19:37:51.523647+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.\u001b[0m\u001b[33m\r\n",
      "\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m25.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m26.2.1\u001b[0m\r\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cloning into 'Atomic_AI_hybrid-v0.1'...\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "remote: Enumerating objects: 322, done.\u001b[K\r\n",
      "remote: Counting objects:   3% (1/28)\u001b[K\r",
      "remote: Counting objects:   7% (2/28)\u001b[K\r",
      "remote: Counting objects:  10% (3/28)\u001b[K\r",
      "remote: Counting objects:  14% (4/28)\u001b[K\r",
      "remote: Counting objects:  17% (5/28)\u001b[K\r",
      "remote: Counting objects:  21% (6/28)\u001b[K\r",
      "remote: Counting objects:  25% (7/28)\u001b[K\r",
      "remote: Counting objects:  28% (8/28)\u001b[K\r",
      "remote: Counting objects:  32% (9/28)\u001b[K\r",
      "remote: Counting objects:  35% (10/28)\u001b[K\r",
      "remote: Counting objects:  39% (11/28)\u001b[K\r",
      "remote: Counting objects:  42% (12/28)\u001b[K\r",
      "remote: Counting objects:  46% (13/28)\u001b[K\r",
      "remote: Counting objects:  50% (14/28)\u001b[K\r",
      "remote: Counting objects:  53% (15/28)\u001b[K\r",
      "remote: Counting objects:  57% (16/28)\u001b[K\r",
      "remote: Counting objects:  60% (17/28)\u001b[K\r",
      "remote: Counting objects:  64% (18/28)\u001b[K\r",
      "remote: Counting objects:  67% (19/28)\u001b[K\r",
      "remote: Counting objects:  71% (20/28)\u001b[K\r",
      "remote: Counting objects:  75% (21/28)\u001b[K\r",
      "remote: Counting objects:  78% (22/28)\u001b[K\r",
      "remote: Counting objects:  82% (23/28)\u001b[K\r",
      "remote: Counting objects:  85% (24/28)\u001b[K\r",
      "remote: Counting objects:  89% (25/28)\u001b[K\r",
      "remote: Counting objects:  92% (26/28)\u001b[K\r",
      "remote: Counting objects:  96% (27/28)\u001b[K\r",
      "remote: Counting objects: 100% (28/28)\u001b[K\r",
      "remote: Counting objects: 100% (28/28), done.\u001b[K\r\n",
      "remote: Compressing objects:   3% (1/28)\u001b[K\r",
      "remote: Compressing objects:   7% (2/28)\u001b[K\r",
      "remote: Compressing objects:  10% (3/28)\u001b[K\r",
      "remote: Compressing objects:  14% (4/28)\u001b[K\r",
      "remote: Compressing objects:  17% (5/28)\u001b[K\r",
      "remote: Compressing objects:  21% (6/28)\u001b[K\r",
      "remote: Compressing objects:  25% (7/28)\u001b[K\r",
      "remote: Compressing objects:  28% (8/28)\u001b[K\r",
      "remote: Compressing objects:  32% (9/28)\u001b[K\r",
      "remote: Compressing objects:  35% (10/28)\u001b[K\r",
      "remote: Compressing objects:  39% (11/28)\u001b[K\r",
      "remote: Compressing objects:  42% (12/28)\u001b[K\r",
      "remote: Compressing objects:  46% (13/28)\u001b[K\r",
      "remote: Compressing objects:  50% (14/28)\u001b[K\r",
      "remote: Compressing objects:  53% (15/28)\u001b[K\r",
      "remote: Compressing objects:  57% (16/28)\u001b[K\r",
      "remote: Compressing objects:  60% (17/28)\u001b[K\r",
      "remote: Compressing objects:  64% (18/28)\u001b[K\r",
      "remote: Compressing objects:  67% (19/28)\u001b[K\r",
      "remote: Compressing objects:  71% (20/28)\u001b[K\r",
      "remote: Compressing objects:  75% (21/28)\u001b[K\r",
      "remote: Compressing objects:  78% (22/28)\u001b[K\r",
      "remote: Compressing objects:  82% (23/28)\u001b[K\r",
      "remote: Compressing objects:  85% (24/28)\u001b[K\r",
      "remote: Compressing objects:  89% (25/28)\u001b[K\r",
      "remote: Compressing objects:  92% (26/28)\u001b[K\r",
      "remote: Compressing objects:  96% (27/28)\u001b[K\r",
      "remote: Compressing objects: 100% (28/28)\u001b[K\r",
      "remote: Compressing objects: 100% (28/28), done.\u001b[K\r\n",
      "Receiving objects:   0% (1/322)\r",
      "Receiving objects:   1% (4/322)\r",
      "Receiving objects:   2% (7/322)\r",
      "Receiving objects:   3% (10/322)\r",
      "Receiving objects:   4% (13/322)\r",
      "Receiving objects:   5% (17/322)\r",
      "Receiving objects:   6% (20/322)\r",
      "Receiving objects:   7% (23/322)\r",
      "Receiving objects:   8% (26/322)\r",
      "Receiving objects:   9% (29/322)\r",
      "Receiving objects:  10% (33/322)\r",
      "Receiving objects:  11% (36/322)\r",
      "Receiving objects:  12% (39/322)\r",
      "Receiving objects:  13% (42/322)\r",
      "Receiving objects:  14% (46/322)\r",
      "Receiving objects:  15% (49/322)\r",
      "Receiving objects:  16% (52/322)\r",
      "Receiving objects:  17% (55/322)\r",
      "Receiving objects:  18% (58/322)\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Receiving objects:  19% (62/322)\r",
      "Receiving objects:  20% (65/322)\r",
      "Receiving objects:  21% (68/322)\r",
      "Receiving objects:  22% (71/322)\r",
      "Receiving objects:  23% (75/322)\r",
      "Receiving objects:  24% (78/322)\r",
      "Receiving objects:  25% (81/322)\r",
      "Receiving objects:  26% (84/322)\r",
      "Receiving objects:  27% (87/322)\r",
      "Receiving objects:  28% (91/322)\r",
      "Receiving objects:  29% (94/322)\r",
      "Receiving objects:  30% (97/322)\r",
      "Receiving objects:  31% (100/322)\r",
      "Receiving objects:  32% (104/322)\r",
      "Receiving objects:  33% (107/322)\r",
      "Receiving objects:  34% (110/322)\r",
      "Receiving objects:  35% (113/322)\r",
      "Receiving objects:  36% (116/322)\r",
      "Receiving objects:  37% (120/322)\r",
      "Receiving objects:  38% (123/322)\r",
      "Receiving objects:  39% (126/322)\r",
      "Receiving objects:  40% (129/322)\r",
      "Receiving objects:  41% (133/322)\r",
      "Receiving objects:  42% (136/322)\r",
      "Receiving objects:  43% (139/322)\r",
      "Receiving objects:  44% (142/322)\r",
      "Receiving objects:  45% (145/322)\r",
      "Receiving objects:  46% (149/322)\r",
      "Receiving objects:  47% (152/322)\r",
      "Receiving objects:  48% (155/322)\r",
      "Receiving objects:  49% (158/322)\r",
      "Receiving objects:  50% (161/322)\r",
      "Receiving objects:  51% (165/322)\r",
      "Receiving objects:  52% (168/322)\r",
      "Receiving objects:  53% (171/322)\r",
      "Receiving objects:  54% (174/322)\r",
      "Receiving objects:  55% (178/322)\r",
      "Receiving objects:  56% (181/322)\r",
      "Receiving objects:  57% (184/322)\r",
      "Receiving objects:  58% (187/322)\r",
      "Receiving objects:  59% (190/322)\r",
      "Receiving objects:  60% (194/322)\r",
      "Receiving objects:  61% (197/322)\r",
      "Receiving objects:  62% (200/322)\r",
      "Receiving objects:  63% (203/322)\r",
      "Receiving objects:  64% (207/322)\r",
      "Receiving objects:  65% (210/322)\r",
      "Receiving objects:  66% (213/322)\r",
      "Receiving objects:  67% (216/322)\r",
      "Receiving objects:  68% (219/322)\r",
      "Receiving objects:  69% (223/322)\r",
      "Receiving objects:  70% (226/322)\r",
      "Receiving objects:  71% (229/322)\r",
      "Receiving objects:  72% (232/322)\r",
      "Receiving objects:  73% (236/322)\r",
      "Receiving objects:  74% (239/322)\r",
      "Receiving objects:  75% (242/322)\r",
      "Receiving objects:  76% (245/322)\r",
      "remote: Total 322 (delta 13), reused 0 (delta 0), pack-reused 294 (from 2)\u001b[K\r\n",
      "Receiving objects:  77% (248/322)\r",
      "Receiving objects:  78% (252/322)\r",
      "Receiving objects:  79% (255/322)\r",
      "Receiving objects:  80% (258/322)\r",
      "Receiving objects:  81% (261/322)\r",
      "Receiving objects:  82% (265/322)\r",
      "Receiving objects:  83% (268/322)\r",
      "Receiving objects:  84% (271/322)\r",
      "Receiving objects:  85% (274/322)\r",
      "Receiving objects:  86% (277/322)\r",
      "Receiving objects:  87% (281/322)\r",
      "Receiving objects:  88% (284/322)\r",
      "Receiving objects:  89% (287/322)\r",
      "Receiving objects:  90% (290/322)\r",
      "Receiving objects:  91% (294/322)\r",
      "Receiving objects:  92% (297/322)\r",
      "Receiving objects:  93% (300/322)\r",
      "Receiving objects:  94% (303/322)\r",
      "Receiving objects:  95% (306/322)\r",
      "Receiving objects:  96% (310/322)\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Receiving objects:  97% (313/322)\r",
      "Receiving objects:  98% (316/322)\r",
      "Receiving objects:  99% (319/322)\r",
      "Receiving objects: 100% (322/322)\r",
      "Receiving objects: 100% (322/322), 207.83 KiB | 2.70 MiB/s, done.\r\n",
      "Resolving deltas:   0% (0/188)\r",
      "Resolving deltas:   1% (2/188)\r",
      "Resolving deltas:   3% (6/188)\r",
      "Resolving deltas:   4% (9/188)\r",
      "Resolving deltas:   5% (10/188)\r",
      "Resolving deltas:   6% (12/188)\r",
      "Resolving deltas:   7% (15/188)\r",
      "Resolving deltas:   8% (16/188)\r",
      "Resolving deltas:   9% (17/188)\r",
      "Resolving deltas:  11% (21/188)\r",
      "Resolving deltas:  13% (26/188)\r",
      "Resolving deltas:  14% (27/188)\r",
      "Resolving deltas:  15% (29/188)\r",
      "Resolving deltas:  16% (31/188)\r",
      "Resolving deltas:  17% (32/188)\r",
      "Resolving deltas:  18% (34/188)\r",
      "Resolving deltas:  19% (36/188)\r",
      "Resolving deltas:  20% (38/188)\r",
      "Resolving deltas:  21% (40/188)\r",
      "Resolving deltas:  22% (43/188)\r",
      "Resolving deltas:  23% (44/188)\r",
      "Resolving deltas:  24% (46/188)\r",
      "Resolving deltas:  25% (47/188)\r",
      "Resolving deltas:  26% (49/188)\r",
      "Resolving deltas:  27% (51/188)\r",
      "Resolving deltas:  28% (53/188)\r",
      "Resolving deltas:  29% (55/188)\r",
      "Resolving deltas:  30% (57/188)\r",
      "Resolving deltas:  31% (59/188)\r",
      "Resolving deltas:  32% (61/188)\r",
      "Resolving deltas:  33% (63/188)\r",
      "Resolving deltas:  34% (64/188)\r",
      "Resolving deltas:  35% (66/188)\r",
      "Resolving deltas:  36% (68/188)\r",
      "Resolving deltas:  37% (71/188)\r",
      "Resolving deltas:  38% (73/188)\r",
      "Resolving deltas:  39% (74/188)\r",
      "Resolving deltas:  40% (76/188)\r",
      "Resolving deltas:  42% (79/188)\r",
      "Resolving deltas:  45% (85/188)\r",
      "Resolving deltas:  46% (88/188)\r",
      "Resolving deltas:  47% (89/188)\r",
      "Resolving deltas:  48% (92/188)\r",
      "Resolving deltas:  50% (94/188)\r",
      "Resolving deltas:  51% (96/188)\r",
      "Resolving deltas:  52% (98/188)\r",
      "Resolving deltas:  53% (100/188)\r",
      "Resolving deltas:  54% (102/188)\r",
      "Resolving deltas:  55% (105/188)\r",
      "Resolving deltas:  56% (106/188)\r",
      "Resolving deltas:  57% (108/188)\r",
      "Resolving deltas:  58% (110/188)\r",
      "Resolving deltas:  59% (111/188)\r",
      "Resolving deltas:  60% (113/188)\r",
      "Resolving deltas:  61% (115/188)\r",
      "Resolving deltas:  62% (117/188)\r",
      "Resolving deltas:  63% (119/188)\r",
      "Resolving deltas:  64% (121/188)\r",
      "Resolving deltas:  68% (128/188)\r",
      "Resolving deltas:  69% (130/188)\r",
      "Resolving deltas:  72% (136/188)\r",
      "Resolving deltas:  73% (138/188)\r",
      "Resolving deltas:  75% (141/188)\r",
      "Resolving deltas:  76% (143/188)\r",
      "Resolving deltas:  77% (145/188)\r",
      "Resolving deltas:  78% (147/188)\r",
      "Resolving deltas:  79% (149/188)\r",
      "Resolving deltas:  80% (151/188)\r",
      "Resolving deltas:  81% (153/188)\r",
      "Resolving deltas:  82% (155/188)\r",
      "Resolving deltas:  83% (157/188)\r",
      "Resolving deltas:  84% (158/188)\r",
      "Resolving deltas:  85% (160/188)\r",
      "Resolving deltas:  86% (162/188)\r",
      "Resolving deltas:  87% (164/188)\r",
      "Resolving deltas:  88% (166/188)\r",
      "Resolving deltas:  89% (168/188)\r",
      "Resolving deltas:  90% (170/188)\r",
      "Resolving deltas:  91% (172/188)\r",
      "Resolving deltas:  92% (173/188)\r",
      "Resolving deltas:  93% (175/188)\r",
      "Resolving deltas:  94% (178/188)\r",
      "Resolving deltas:  95% (180/188)\r",
      "Resolving deltas:  96% (182/188)\r",
      "Resolving deltas:  97% (183/188)\r",
      "Resolving deltas:  98% (185/188)\r",
      "Resolving deltas:  99% (187/188)\r",
      "Resolving deltas: 100% (188/188)\r",
      "Resolving deltas: 100% (188/188), done.\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: jax[tpu] in /usr/local/lib/python3.12/site-packages (0.10.2)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting jax[tpu]\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Downloading jax-0.11.0-py3-none-any.whl.metadata (13 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting jaxlib<=0.11.0,>=0.11.0 (from jax[tpu])\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Downloading jaxlib-0.11.0-cp312-cp312-manylinux_2_27_x86_64.whl.metadata (1.3 kB)\r\n",
      "Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (0.5.4)\r\n",
      "Requirement already satisfied: numpy>=2.1 in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (2.5.0)\r\n",
      "Requirement already satisfied: opt_einsum in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (3.4.0)\r\n",
      "Requirement already satisfied: scipy>=1.15 in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (1.18.0)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting libtpu==0.0.44.* (from jax[tpu])\r\n",
      "  Downloading libtpu-0.0.44.1-cp312-cp312-manylinux_2_31_x86_64.whl.metadata (1.5 kB)\r\n",
      "Requirement already satisfied: requests in /usr/local/lib/python3.12/site-packages (from jax[tpu]) (2.34.2)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (3.4.7)\r\n",
      "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (3.18)\r\n",
      "Requirement already satisfied: urllib3<3,>=1.26 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (2.7.0)\r\n",
      "Requirement already satisfied: certifi>=2023.5.7 in /usr/local/lib/python3.12/site-packages (from requests->jax[tpu]) (2026.6.17)\r\n",
      "Downloading libtpu-0.0.44.1-cp312-cp312-manylinux_2_31_x86_64.whl (215.6 MB)\r\n",
      "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/215.6 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m40.1/215.6 MB\u001b[0m \u001b[31m232.2 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m96.5/215.6 MB\u001b[0m \u001b[31m258.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━\u001b[0m \u001b[32m141.6/215.6 MB\u001b[0m \u001b[31m246.1 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━\u001b[0m \u001b[32m187.4/215.6 MB\u001b[0m \u001b[31m244.5 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m215.5/215.6 MB\u001b[0m \u001b[31m232.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m215.6/215.6 MB\u001b[0m \u001b[31m44.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25h"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading jaxlib-0.11.0-cp312-cp312-manylinux_2_27_x86_64.whl (87.3 MB)\r\n",
      "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/87.3 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m30.4/87.3 MB\u001b[0m \u001b[31m152.8 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━\u001b[0m \u001b[32m53.7/87.3 MB\u001b[0m \u001b[31m133.8 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━\u001b[0m \u001b[32m78.6/87.3 MB\u001b[0m \u001b[31m130.3 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m87.0/87.3 MB\u001b[0m \u001b[31m129.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m87.0/87.3 MB\u001b[0m \u001b[31m129.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m87.0/87.3 MB\u001b[0m \u001b[31m129.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m87.0/87.3 MB\u001b[0m \u001b[31m129.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m87.0/87.3 MB\u001b[0m \u001b[31m129.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m87.3/87.3 MB\u001b[0m \u001b[31m49.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading jax-0.11.0-py3-none-any.whl (3.3 MB)\r\n",
      "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/3.3 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━\u001b[0m \u001b[32m3.1/3.3 MB\u001b[0m \u001b[31m136.0 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━\u001b[0m \u001b[32m3.1/3.3 MB\u001b[0m \u001b[31m136.0 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━\u001b[0m \u001b[32m3.1/3.3 MB\u001b[0m \u001b[31m136.0 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━\u001b[0m \u001b[32m3.1/3.3 MB\u001b[0m \u001b[31m136.0 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m3.3/3.3 MB\u001b[0m \u001b[31m3.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25h"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Installing collected packages: libtpu, jaxlib, jax\r\n",
      "  Attempting uninstall: libtpu\r\n",
      "    Found existing installation: libtpu 0.0.17\r\n",
      "    Uninstalling libtpu-0.0.17:\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Successfully uninstalled libtpu-0.0.17\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Attempting uninstall: jaxlib\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Found existing installation: jaxlib 0.10.2\r\n",
      "    Uninstalling jaxlib-0.10.2:\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Successfully uninstalled jaxlib-0.10.2\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Attempting uninstall: jax\r\n",
      "    Found existing installation: jax 0.10.2\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Uninstalling jax-0.10.2:\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Successfully uninstalled jax-0.10.2\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Successfully installed jax-0.11.0 jaxlib-0.11.0 libtpu-0.0.44.1\r\n",
      "\u001b[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.\u001b[0m\u001b[33m\r\n",
      "\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m25.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m26.2.1\u001b[0m\r\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\r\n"
     ]
    }
   ],
   "source": [
    "!pip install -q huggingface_hub\n",
    "!rm -rf Atomic_AI_hybrid-v0.1\n",
    "!git clone https://github.com/Akseleu-J/Atomic_AI_hybrid-v0.1.git\n",
    "!pip install -U \"jax[tpu]\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "300e75ef",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-08T19:38:25.425259Z",
     "iopub.status.busy": "2026-08-08T19:38:25.425035Z",
     "iopub.status.idle": "2026-08-08T19:38:25.428127Z",
     "shell.execute_reply": "2026-08-08T19:38:25.427496Z"
    },
    "papermill": {
     "duration": 0.008978,
     "end_time": "2026-08-08T19:38:25.428682+00:00",
     "exception": false,
     "start_time": "2026-08-08T19:38:25.419704+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import sys, os\n",
    "sys.path.append(\"Atomic_AI_hybrid-v0.1\")\n",
    "os.environ[\"HF_REPO_ID\"] = \"atomic-ai-labs/atomic-light-v0.1\"\n",
    "# HF_TOKEN уже в Kaggle Secrets\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a435eb5d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-08T19:38:25.439227Z",
     "iopub.status.busy": "2026-08-08T19:38:25.439076Z",
     "iopub.status.idle": "2026-08-09T04:00:49.858981Z",
     "shell.execute_reply": "2026-08-09T04:00:49.857705Z"
    },
    "papermill": {
     "duration": 30144.427067,
     "end_time": "2026-08-09T04:00:49.860053+00:00",
     "exception": false,
     "start_time": "2026-08-08T19:38:25.432986+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.12/site-packages/jax/_src/cloud_tpu_init.py:88: UserWarning: Transparent hugepages are not enabled. TPU runtime startup and shutdown time should be significantly improved on TPU v5e and newer. If not already set, you may need to enable transparent hugepages in your VM image (sudo sh -c \"echo always > /sys/kernel/mm/transparent_hugepage/enabled\")\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Интеграция: atomic-ai-labs/atomic-light-v0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Logging before InitGoogle() is written to STDERR\n",
      "E0000 00:00:1786217913.624548      73 common_lib.cc:943] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8471 in any of the 0 ports provided in `tpu_process_addresses`=\"local\"\n",
      "=== Source Location Trace: ===\n",
      "learning/45eac/tfrc/runtime/common_lib.cc:239\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Configured `CheckpointManager` using deprecated legacy API. Please follow the instructions at https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html to migrate.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Configured `CheckpointManager` using deprecated legacy API. Please follow the instructions at https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html to migrate.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Configured `CheckpointManager` using deprecated legacy API. Please follow the instructions at https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html to migrate.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] Downloading slot 'latest' from atomic-ai-labs/atomic-light-v0.1...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "922d8ca2726a473a85955edecd95b464",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading (incomplete total...): 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d9c8b5559bae4940b47dec1c00315e54",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Fetching 16 files:   0%|          | 0/16 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Configured `CheckpointManager` using deprecated legacy API. Please follow the instructions at https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html to migrate.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] Slot 'latest': найден шаг 94\n",
      "Все файлы найдены.\n",
      "[DATA] codex_input_ids.npy: 331,776 блоков\n",
      "[DATA] kodcode_input_ids.npy: 28,375 блоков\n",
      "[DATA] math_input_ids.npy: 73,728 блоков\n",
      "[DATA] rstar_input_ids.npy: 245,760 блоков\n",
      "[DATA] syntheticcode_input_ids.npy: 88,497 блоков\n",
      "[DATA] Комбинированный пул: 768,136 блоков из 5 файл(ов)\n",
      "Всего блоков (полный пул): 768,136\n",
      "Всего блоков (после 100% подвыборки): 768,136\n",
      "[TPU] Компиляция XLA графа под 22804 эффективных шагов (1 эпох(и) x 22804 шагов, accum=4)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TPU] Устройств в mesh: 8 (FSDP: params, state и батч шардированы).\n",
      "[DATA] codex_input_ids.npy: 331,776 блоков\n",
      "[DATA] kodcode_input_ids.npy: 28,375 блоков\n",
      "[DATA] math_input_ids.npy: 73,728 блоков\n",
      "[DATA] rstar_input_ids.npy: 245,760 блоков\n",
      "[DATA] syntheticcode_input_ids.npy: 88,497 блоков\n",
      "[DATA] Комбинированный пул: 768,136 блоков из 5 файл(ов)\n",
      "[SANITY] Проверка первого батча...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[SANITY] Labels range: [0, 128012], vocab_size=128256\n",
      "[SANITY] Валидных labels в батче: 32768/32768 (100.0%)\n",
      "[SANITY] labels[i]==ids[i+1] (должно быть высоким): 100.00%\n",
      "[SANITY] labels[i]==ids[i]   (должно быть низким): 0.20%\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[MEM] Доступно памяти на чипе 0: {'num_allocs': 486, 'bytes_in_use': 1961653248, 'peak_bytes_in_use': 1961653248, 'largest_alloc_size': 100663296, 'bytes_limit': 16909332480, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'bytes_reservable_limit': 16909332480, 'largest_free_block_bytes': 14947679232}\n",
      "Общее количество параметров: 581,553,280 (≈ 0.58 млрд)\n",
      "Размер весов модели (глобально): 2.33 ГБ (с FSDP на чип реально хранится в среднем ~0.29 ГБ)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[RESUME] ⬆️ Restoring step 94 из 'latest'...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[RESUME DEBUG] param_norm=765.9128, has_nan=False\n",
      "[RESUME] ✅ Restored: step=94, best_val=inf, best_train=inf\n",
      "[DATA] codex_input_ids.npy: 331,776 блоков\n",
      "[DATA] kodcode_input_ids.npy: 28,375 блоков\n",
      "[DATA] math_input_ids.npy: 73,728 блоков\n",
      "[DATA] rstar_input_ids.npy: 245,760 блоков\n",
      "[DATA] syntheticcode_input_ids.npy: 88,497 блоков\n",
      "[DATA] Комбинированный пул: 768,136 блоков из 5 файл(ов)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[MEM ANALYSIS] HBM temp:      7.20 ГБ\n",
      "[MEM ANALYSIS] HBM arguments: 3.98 ГБ\n",
      "[MEM ANALYSIS] HBM output:    3.98 ГБ\n",
      "[TPU] Компиляция готова -- переходим к реальному обучению.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[DATA] Resume: пропускаем первые 376 микрошагов текущего прохода датасета (уже были пройдены раньше).\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 0: данные=100мс  TPU=3987мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 1: данные=83мс  TPU=3909мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 2: данные=74мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] effective step 1: данные=64мс  TPU compute=3908мс  apply=41763мс  (доля данных: 0%)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 4: данные=65мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 5: данные=63мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 6: данные=67мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] effective step 2: данные=70мс  TPU compute=3908мс  apply=70мс  (доля данных: 2%)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 8: данные=69мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 9: данные=64мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 10: данные=67мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] effective step 3: данные=68мс  TPU compute=3907мс  apply=70мс  (доля данных: 2%)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 12: данные=74мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 13: данные=66мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 14: данные=85мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] effective step 4: данные=62мс  TPU compute=3908мс  apply=69мс  (доля данных: 2%)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 16: данные=75мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 17: данные=76мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 18: данные=65мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] effective step 5: данные=66мс  TPU compute=3908мс  apply=71мс  (доля данных: 2%)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 20: данные=71мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 21: данные=73мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 22: данные=65мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] effective step 6: данные=67мс  TPU compute=3908мс  apply=70мс  (доля данных: 2%)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 24: данные=76мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 25: данные=186мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 26: данные=106мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] effective step 7: данные=67мс  TPU compute=3908мс  apply=71мс  (доля данных: 2%)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 28: данные=72мс  TPU=3908мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[TIMING] micro step 29: данные=64мс  TPU=3907мс  (accumulating)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 10/22804 | Global Step: 104 | Train Loss: 6.7129 (ce=6.7129 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 20/22804 | Global Step: 114 | Train Loss: 6.3150 (ce=6.3150 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 30/22804 | Global Step: 124 | Train Loss: 6.1431 (ce=6.1431 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 40/22804 | Global Step: 134 | Train Loss: 5.7365 (ce=5.7365 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 50/22804 | Global Step: 144 | Train Loss: 5.3402 (ce=5.3402 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 60/22804 | Global Step: 154 | Train Loss: 5.7423 (ce=5.7423 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 70/22804 | Global Step: 164 | Train Loss: 5.6993 (ce=5.6993 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 80/22804 | Global Step: 174 | Train Loss: 5.2920 (ce=5.2920 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 90/22804 | Global Step: 184 | Train Loss: 5.2694 (ce=5.2694 aux=0.0000 z=0.00000) | best_train=inf\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 189 (прошло 25.2 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/189 (заняло 53.6с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/189\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/best_train/189 (заняло 31.2с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: best_train/189\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [best_train] удалён старый шаг: 94\n",
      "[BEST_TRAIN] Новый лучший train_loss: 5.4800 на шаге 189\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 100/22804 | Global Step: 194 | Train Loss: 5.2645 (ce=5.2645 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 110/22804 | Global Step: 204 | Train Loss: 5.0595 (ce=5.0595 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 120/22804 | Global Step: 214 | Train Loss: 5.0541 (ce=5.0541 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 130/22804 | Global Step: 224 | Train Loss: 4.8641 (ce=4.8641 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 140/22804 | Global Step: 234 | Train Loss: 4.7958 (ce=4.7958 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 150/22804 | Global Step: 244 | Train Loss: 4.6527 (ce=4.6527 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 160/22804 | Global Step: 254 | Train Loss: 4.6632 (ce=4.6632 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 170/22804 | Global Step: 264 | Train Loss: 4.4503 (ce=4.4503 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 180/22804 | Global Step: 274 | Train Loss: 4.2868 (ce=4.2868 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 190/22804 | Global Step: 284 | Train Loss: 4.5158 (ce=4.5158 aux=0.0000 z=0.00000) | best_train=5.4800\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 30.800729 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 286 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/286 (заняло 50.1с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/286\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 28.934961 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 94\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/best_train/286 (заняло 28.9с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: best_train/286\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [best_train] удалён старый шаг: 189\n",
      "[BEST_TRAIN] Новый лучший train_loss: 4.5211 на шаге 286\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 200/22804 | Global Step: 294 | Train Loss: 4.4901 (ce=4.4901 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[EVAL] Step 300: val loss (частичный, 40 батчей) = 11.7618\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/best_val/300 (заняло 40.9с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: best_val/300\n",
      "[BEST_VAL] Новый лучший val_loss: 11.7618 на шаге 300\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 210/22804 | Global Step: 304 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 220/22804 | Global Step: 314 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 230/22804 | Global Step: 324 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 240/22804 | Global Step: 334 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 250/22804 | Global Step: 344 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 260/22804 | Global Step: 354 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 270/22804 | Global Step: 364 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 280/22804 | Global Step: 374 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 29.200560 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 377 (прошло 25.2 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/377 (заняло 176.4с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/377\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 189\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 290/22804 | Global Step: 384 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 300/22804 | Global Step: 394 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 310/22804 | Global Step: 404 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 320/22804 | Global Step: 414 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 330/22804 | Global Step: 424 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 340/22804 | Global Step: 434 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 350/22804 | Global Step: 444 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 360/22804 | Global Step: 454 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 370/22804 | Global Step: 464 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 154.987831 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 474 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/474 (заняло 178.4с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/474\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 286\n",
      "Epoch: 0 | Step: 380/22804 | Global Step: 474 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 390/22804 | Global Step: 484 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 400/22804 | Global Step: 494 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 410/22804 | Global Step: 504 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 420/22804 | Global Step: 514 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 430/22804 | Global Step: 524 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 440/22804 | Global Step: 534 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 450/22804 | Global Step: 544 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 460/22804 | Global Step: 554 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 470/22804 | Global Step: 564 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 156.920763 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 571 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/571 (заняло 178.3с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/571\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 377\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 480/22804 | Global Step: 574 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 490/22804 | Global Step: 584 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 500/22804 | Global Step: 594 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[EVAL] Step 600: val loss (частичный, 40 батчей) = 11.7618\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 510/22804 | Global Step: 604 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 520/22804 | Global Step: 614 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 530/22804 | Global Step: 624 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 540/22804 | Global Step: 634 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 550/22804 | Global Step: 644 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 560/22804 | Global Step: 654 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 570/22804 | Global Step: 664 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 156.839975 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 665 (прошло 25.1 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/665 (заняло 166.5с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/665\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 474\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 580/22804 | Global Step: 674 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 590/22804 | Global Step: 684 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 600/22804 | Global Step: 694 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 610/22804 | Global Step: 704 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 620/22804 | Global Step: 714 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 630/22804 | Global Step: 724 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 640/22804 | Global Step: 734 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 650/22804 | Global Step: 744 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 660/22804 | Global Step: 754 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 149.903491 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 762 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/762 (заняло 184.8с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/762\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 571\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 670/22804 | Global Step: 764 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 680/22804 | Global Step: 774 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 690/22804 | Global Step: 784 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 700/22804 | Global Step: 794 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 710/22804 | Global Step: 804 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 720/22804 | Global Step: 814 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 730/22804 | Global Step: 824 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 740/22804 | Global Step: 834 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 750/22804 | Global Step: 844 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 760/22804 | Global Step: 854 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 163.926010 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 859 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/859 (заняло 172.3с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/859\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 665\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 770/22804 | Global Step: 864 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 780/22804 | Global Step: 874 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 790/22804 | Global Step: 884 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 800/22804 | Global Step: 894 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[EVAL] Step 900: val loss (частичный, 40 батчей) = 11.7618\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 810/22804 | Global Step: 904 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 820/22804 | Global Step: 914 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 830/22804 | Global Step: 924 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 840/22804 | Global Step: 934 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 850/22804 | Global Step: 944 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 151.381595 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 953 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/953 (заняло 178.9с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/953\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 762\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 860/22804 | Global Step: 954 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 870/22804 | Global Step: 964 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 880/22804 | Global Step: 974 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 890/22804 | Global Step: 984 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 900/22804 | Global Step: 994 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 910/22804 | Global Step: 1004 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 920/22804 | Global Step: 1014 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 930/22804 | Global Step: 1024 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 940/22804 | Global Step: 1034 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 950/22804 | Global Step: 1044 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 157.458150 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1050 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1050 (заняло 169.9с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1050\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 859\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 960/22804 | Global Step: 1054 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 970/22804 | Global Step: 1064 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 980/22804 | Global Step: 1074 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 990/22804 | Global Step: 1084 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1000/22804 | Global Step: 1094 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1010/22804 | Global Step: 1104 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1020/22804 | Global Step: 1114 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1030/22804 | Global Step: 1124 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1040/22804 | Global Step: 1134 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1050/22804 | Global Step: 1144 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 148.524693 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1147 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1147 (заняло 169.5с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1147\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 953\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1060/22804 | Global Step: 1154 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1070/22804 | Global Step: 1164 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1080/22804 | Global Step: 1174 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1090/22804 | Global Step: 1184 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1100/22804 | Global Step: 1194 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[EVAL] Step 1200: val loss (частичный, 40 батчей) = 11.7618\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1110/22804 | Global Step: 1204 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1120/22804 | Global Step: 1214 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1130/22804 | Global Step: 1224 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1140/22804 | Global Step: 1234 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 148.559140 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1241 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1241 (заняло 164.4с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1241\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 1050\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1150/22804 | Global Step: 1244 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1160/22804 | Global Step: 1254 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1170/22804 | Global Step: 1264 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1180/22804 | Global Step: 1274 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1190/22804 | Global Step: 1284 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1200/22804 | Global Step: 1294 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1210/22804 | Global Step: 1304 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1220/22804 | Global Step: 1314 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1230/22804 | Global Step: 1324 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1240/22804 | Global Step: 1334 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 143.495409 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1338 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1338 (заняло 173.7с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1338\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 1147\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1250/22804 | Global Step: 1344 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1260/22804 | Global Step: 1354 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1270/22804 | Global Step: 1364 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1280/22804 | Global Step: 1374 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1290/22804 | Global Step: 1384 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1300/22804 | Global Step: 1394 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1310/22804 | Global Step: 1404 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1320/22804 | Global Step: 1414 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1330/22804 | Global Step: 1424 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1340/22804 | Global Step: 1434 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 152.764817 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1435 (прошло 25.1 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1435 (заняло 173.5с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1435\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 1241\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1350/22804 | Global Step: 1444 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1360/22804 | Global Step: 1454 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1370/22804 | Global Step: 1464 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1380/22804 | Global Step: 1474 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1390/22804 | Global Step: 1484 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1400/22804 | Global Step: 1494 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[EVAL] Step 1500: val loss (частичный, 40 батчей) = 11.7618\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1410/22804 | Global Step: 1504 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1420/22804 | Global Step: 1514 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1430/22804 | Global Step: 1524 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 156.135486 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1529 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1529 (заняло 176.8с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1529\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 1338\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1440/22804 | Global Step: 1534 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1450/22804 | Global Step: 1544 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1460/22804 | Global Step: 1554 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1470/22804 | Global Step: 1564 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1480/22804 | Global Step: 1574 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1490/22804 | Global Step: 1584 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1500/22804 | Global Step: 1594 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1510/22804 | Global Step: 1604 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1520/22804 | Global Step: 1614 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1530/22804 | Global Step: 1624 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 156.279180 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1626 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1626 (заняло 179.2с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1626\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 1435\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1540/22804 | Global Step: 1634 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1550/22804 | Global Step: 1644 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1560/22804 | Global Step: 1654 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1570/22804 | Global Step: 1664 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1580/22804 | Global Step: 1674 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1590/22804 | Global Step: 1684 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1600/22804 | Global Step: 1694 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1610/22804 | Global Step: 1704 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1620/22804 | Global Step: 1714 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 158.291479 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] 💾 Цикл сохранения на шаге 1723 (прошло 25.0 мин)...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1723 (заняло 182.8с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1723\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 1529\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1630/22804 | Global Step: 1724 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1640/22804 | Global Step: 1734 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1650/22804 | Global Step: 1744 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1660/22804 | Global Step: 1754 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1670/22804 | Global Step: 1764 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1680/22804 | Global Step: 1774 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1690/22804 | Global Step: 1784 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0 | Step: 1700/22804 | Global Step: 1794 | Train Loss: 11.7618 (ce=11.7618 aux=0.0000 z=0.00000) | best_train=4.5211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Waiting for previous save to complete took 161.839984 seconds. If this number is high, consider checkpointing less frequently.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[EVAL] Step 1800: val loss (частичный, 40 батчей) = 11.7618\n",
      "[EARLY STOP] Частичный val loss не улучшался 5 проверок подряд. Останавливаю обучение немедленно.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CKPT] Сохранён локально: /kaggle/working/orbax_checkpoints/latest/1800 (заняло 164.0с)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] ✅ Uploaded: latest/1800\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[HF] 🗑️ [latest] удалён старый шаг: 1626\n",
      "[ORBAX] Финальные чекпоинты (шаг 1800) сохранены.\n",
      "Обучение завершено (для этой сессии).\n"
     ]
    }
   ],
   "source": [
    "import glob\n",
    "import os\n",
    "import re\n",
    "import time\n",
    "import json\n",
    "import signal\n",
    "import sys\n",
    "\n",
    "import jax\n",
    "import jax.numpy as jnp\n",
    "import numpy as np\n",
    "import optax\n",
    "import orbax.checkpoint as ocp\n",
    "from jax.experimental import mesh_utils\n",
    "from jax.sharding import Mesh, NamedSharding\n",
    "from jax.sharding import PartitionSpec as P\n",
    "\n",
    "# ==================== HF HUB RELAY ====================\n",
    "try:\n",
    "    from kaggle_secrets import UserSecretsClient\n",
    "    user_secrets = UserSecretsClient()\n",
    "    from huggingface_hub import HfApi, snapshot_download, upload_folder, create_repo, login\n",
    "    HF_TOKEN = user_secrets.get_secret(\"HF_TOKEN\")\n",
    "    HF_REPO_ID = user_secrets.get_secret(\"HF_REPO_ID\")\n",
    "    _HAS_HF = bool(HF_TOKEN)\n",
    "    login(HF_TOKEN)\n",
    "    if _HAS_HF:\n",
    "        print(f\"[HF] ✅ Интеграция: {HF_REPO_ID}\")\n",
    "    else:\n",
    "        raise ImportError(\"Hugging face worck's uncorrectly\")\n",
    "except ImportError:\n",
    "    _HAS_HF = False\n",
    "    print(\"[WARN] pip install -q huggingface_hub\")\n",
    "\n",
    "# ФИКС: раз в 25 минут. ВАЖНО: с синхронным чекпоинтингом (см. ниже) реальная\n",
    "# длительность записи будет видна в логах как время выполнения save_all_slots() --\n",
    "# следите за первыми 2-3 циклами и увеличьте интервал, если запись занимает\n",
    "# больше половины интервала (иначе TPU будет простаивать в ожидании I/O больше,\n",
    "# чем считать).\n",
    "CHECKPOINT_EVERY_SECONDS = 25 * 60\n",
    "\n",
    "SESSION_TIME_BUDGET_SECONDS = 9 * 3600 - 15 * 60  # 9 часов минус запас на graceful stop\n",
    "\n",
    "# ФИКС: 4 именованных слота на HF вместо \"последние N\" -- защищает от того,\n",
    "# что один плохой шаг (как на 414-м) перезатирает единственную сохранённую\n",
    "# копию. Слоты:\n",
    "#   latest      -- держит 2 последних чекпоинта (N и N-1), для обычного resume\n",
    "#   best_train  -- лучший train_loss за всё время\n",
    "#   best_val    -- лучший val_loss (по частичной или полной валидации)\n",
    "HF_LATEST_KEEP_N = 2\n",
    "\n",
    "DATASET_FRACTION = 1\n",
    "DATASET_FRACTION_SEED = 777\n",
    "\n",
    "\n",
    "# ==========================================================================\n",
    "# ФИКС от гонки на шаге 414: enable_async_checkpointing=False.\n",
    "# Async-сохранение копирует params/opt_state с device на host в ФОНОВОМ\n",
    "# потоке и возвращает управление сразу; следующий compiled_apply() при этом\n",
    "# донирует (donate_argnums) те же самые буферы памяти под перезапись. Если\n",
    "# фоновый writer не успел дочитать буфер до того, как XLA его переиспользовал\n",
    "# (см. лог \"Waiting for previous save to complete took 256s\" -- явный признак\n",
    "# отставания фонового воркера), live-параметры после этого момента портятся\n",
    "# необратимо. Синхронный save() блокирует до полного завершения записи, что\n",
    "# делает эту гонку структурно невозможной ценой простоя TPU во время записи.\n",
    "# ==========================================================================\n",
    "\n",
    "def make_manager(local_dir, max_to_keep):\n",
    "    os.makedirs(local_dir, exist_ok=True)\n",
    "    options = ocp.CheckpointManagerOptions(\n",
    "        max_to_keep=max_to_keep,\n",
    "        create=True,\n",
    "        enable_async_checkpointing=False,\n",
    "    )\n",
    "    return ocp.CheckpointManager(local_dir, ocp.StandardCheckpointer(), options)\n",
    "\n",
    "\n",
    "def save_slot(mngr, local_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, train_loss=None):\n",
    "    t0 = time.perf_counter()\n",
    "    mngr.save(step, args=ocp.args.StandardSave({\"params\": params, \"opt_state\": opt_state}))\n",
    "    mngr.wait_until_finished()\n",
    "    elapsed = time.perf_counter() - t0\n",
    "    meta = {\n",
    "        \"global_step\": int(step),\n",
    "        \"epoch\": int(epoch),\n",
    "        \"best_val_loss\": float(best_val_loss),\n",
    "        \"best_train_loss\": float(best_train_loss),\n",
    "        \"timestamp\": time.time(),\n",
    "    }\n",
    "    if train_loss is not None:\n",
    "        meta[\"train_loss\"] = float(jax.device_get(train_loss))\n",
    "    meta_path = os.path.join(local_dir, str(step), \"metadata.json\")\n",
    "    with open(meta_path, \"w\") as f:\n",
    "        json.dump(meta, f)\n",
    "    print(f\"[CKPT] Сохранён локально: {local_dir}/{step} (заняло {elapsed:.1f}с)\")\n",
    "    return elapsed\n",
    "\n",
    "\n",
    "def upload_slot(local_dir, repo_subdir, step, msg=\"\", keep_last_n=1):\n",
    "    \"\"\"Заливает {local_dir}/{step} -> HF под path_in_repo={repo_subdir}/{step},\n",
    "    затем чистит старые шаги ИМЕННО внутри этого repo_subdir (не трогая\n",
    "    другие слоты).\"\"\"\n",
    "    if not _HAS_HF:\n",
    "        return\n",
    "    step_dir = os.path.join(local_dir, str(step))\n",
    "    if not os.path.exists(step_dir):\n",
    "        print(f\"[HF] ⚠️ {step_dir} не найден, пропускаю upload\")\n",
    "        return\n",
    "    try:\n",
    "        api = HfApi(token=HF_TOKEN)\n",
    "        create_repo(HF_REPO_ID, repo_type=\"model\", exist_ok=True)\n",
    "        st_path = os.path.join(step_dir, \"STATUS.txt\")\n",
    "        with open(st_path, \"w\") as f:\n",
    "            f.write(f\"IDLE: slot={repo_subdir} last_step={step} | t={time.time()}\\n\")\n",
    "        upload_folder(\n",
    "            folder_path=step_dir,\n",
    "            repo_id=HF_REPO_ID,\n",
    "            repo_type=\"model\",\n",
    "            path_in_repo=f\"{repo_subdir}/{step}\",\n",
    "            commit_message=f\"[{repo_subdir}] step {step} {msg}\",\n",
    "        )\n",
    "        print(f\"[HF] ✅ Uploaded: {repo_subdir}/{step}\")\n",
    "\n",
    "        try:\n",
    "            all_files = api.list_repo_files(HF_REPO_ID, repo_type=\"model\")\n",
    "            prefix = f\"{repo_subdir}/\"\n",
    "            found_steps = set()\n",
    "            for f_path in all_files:\n",
    "                if f_path.startswith(prefix):\n",
    "                    rest = f_path[len(prefix):]\n",
    "                    m = re.match(r\"^(\\d+)/\", rest)\n",
    "                    if m:\n",
    "                        found_steps.add(int(m.group(1)))\n",
    "            steps_sorted = sorted(found_steps, reverse=True)\n",
    "            for old_step in steps_sorted[keep_last_n:]:\n",
    "                try:\n",
    "                    api.delete_folder(\n",
    "                        path_in_repo=f\"{repo_subdir}/{old_step}\",\n",
    "                        repo_id=HF_REPO_ID,\n",
    "                        repo_type=\"model\",\n",
    "                    )\n",
    "                    print(f\"[HF] 🗑️ [{repo_subdir}] удалён старый шаг: {old_step}\")\n",
    "                except Exception as e_del:\n",
    "                    print(f\"[HF] ⚠️ Не удалось удалить {repo_subdir}/{old_step}: {e_del}\")\n",
    "        except Exception as e_list:\n",
    "            print(f\"[HF] ⚠️ Не удалось получить список файлов для чистки {repo_subdir}: {e_list}\")\n",
    "    except Exception as e:\n",
    "        print(f\"[HF] ❌ Upload error ({repo_subdir}): {e}\")\n",
    "\n",
    "\n",
    "def download_slot(local_dir, repo_subdir):\n",
    "    \"\"\"Скачивает все шаги указанного слота с HF в local_dir. Возвращает\n",
    "    максимальный найденный локально номер шага (или None).\"\"\"\n",
    "    if not _HAS_HF:\n",
    "        return None\n",
    "    try:\n",
    "        print(f\"[HF] Downloading slot '{repo_subdir}' from {HF_REPO_ID}...\")\n",
    "        os.makedirs(local_dir, exist_ok=True)\n",
    "        snapshot_download(\n",
    "            repo_id=HF_REPO_ID,\n",
    "            local_dir=local_dir,\n",
    "            repo_type=\"model\",\n",
    "            allow_patterns=[f\"{repo_subdir}/**\"],\n",
    "        )\n",
    "        # snapshot_download кладёт файлы как local_dir/{repo_subdir}/{step}/... --\n",
    "        # переносим на плоскую структуру local_dir/{step}/..., которую ожидает\n",
    "        # CheckpointManager.\n",
    "        src_root = os.path.join(local_dir, repo_subdir)\n",
    "        if not os.path.isdir(src_root):\n",
    "            return None\n",
    "        for step_name in os.listdir(src_root):\n",
    "            src = os.path.join(src_root, step_name)\n",
    "            dst = os.path.join(local_dir, step_name)\n",
    "            if os.path.isdir(src) and not os.path.exists(dst):\n",
    "                os.rename(src, dst)\n",
    "        items = [d for d in os.listdir(local_dir) if d.isdigit()]\n",
    "        if not items:\n",
    "            return None\n",
    "        latest = max(int(d) for d in items)\n",
    "        print(f\"[HF] Slot '{repo_subdir}': найден шаг {latest}\")\n",
    "        return latest\n",
    "    except Exception as e:\n",
    "        print(f\"[HF] Download failed для слота '{repo_subdir}': {e}\")\n",
    "        return None\n",
    "\n",
    "\n",
    "from model import FullHybridMoEModel, ModelConfig, set_model_mesh, get_model_mesh\n",
    "from optimizer import compute_loss, make_hybrid_optimizer\n",
    "from utils import path_to_str\n",
    "\n",
    "\n",
    "def make_tpu_mesh():\n",
    "    devices = jax.devices()\n",
    "    n = len(devices)\n",
    "    mesh_devices = mesh_utils.create_device_mesh((n,), devices)\n",
    "    return Mesh(mesh_devices, axis_names=(\"tpu_nodes\",))\n",
    "\n",
    "\n",
    "def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int,\n",
    "                           seq_len: int = 8192, accum_steps: int = 1):\n",
    "    mesh = make_tpu_mesh()\n",
    "    n_devices = mesh.shape[\"tpu_nodes\"]\n",
    "\n",
    "    if batch_size % n_devices != 0:\n",
    "        raise ValueError(\n",
    "            f\"batch_size={batch_size} must be divisible by n_devices={n_devices}.\"\n",
    "        )\n",
    "\n",
    "    batch_axis = \"tpu_nodes\"\n",
    "    set_model_mesh(mesh, batch_axis=batch_axis)\n",
    "\n",
    "    tx = make_hybrid_optimizer(total_steps=total_steps)\n",
    "    model = FullHybridMoEModel(cfg=config)\n",
    "\n",
    "    init_rng = jax.random.PRNGKey(0)\n",
    "    abstract_params = jax.eval_shape(\n",
    "        lambda: model.init(init_rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))\n",
    "    )[\"params\"]\n",
    "\n",
    "    data_sharding = NamedSharding(mesh, P(\"tpu_nodes\", None))\n",
    "\n",
    "    MIN_SHARD_SIZE = 128\n",
    "\n",
    "    def _get_shard_spec(path, param):\n",
    "        if not hasattr(param, \"shape\") or param.ndim == 0:\n",
    "            return NamedSharding(mesh, P())\n",
    "        if \"experts_block\" in path_to_str(path):\n",
    "            return NamedSharding(mesh, P(*([None] * param.ndim)))\n",
    "        best_axis, best_size = None, -1\n",
    "        for i, size in enumerate(param.shape):\n",
    "            if size % n_devices == 0 and (size // n_devices) >= MIN_SHARD_SIZE and size > best_size:\n",
    "                best_axis, best_size = i, size\n",
    "        if best_axis is None:\n",
    "            return NamedSharding(mesh, P(*([None] * param.ndim)))\n",
    "        spec = [None] * param.ndim\n",
    "        spec[best_axis] = \"tpu_nodes\"\n",
    "        return NamedSharding(mesh, P(*spec))\n",
    "\n",
    "    param_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, abstract_params)\n",
    "\n",
    "    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))\n",
    "    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, opt_state_abstract)\n",
    "\n",
    "    def model_apply_wrapped(variables, input_ids, rngs=None, deterministic=True, **kwargs):\n",
    "        return model.apply(\n",
    "            variables, input_ids,\n",
    "            rngs=rngs, deterministic=deterministic,\n",
    "            **kwargs\n",
    "        )\n",
    "\n",
    "    def distributed_train_step_micro(p, s, b, r, accum_grads):\n",
    "        def loss_fn(param):\n",
    "            return compute_loss(\n",
    "                param, model_apply_wrapped, b, config,\n",
    "                rngs={\"dropout\": r},\n",
    "                deterministic=False, return_aux=True,\n",
    "                ce_chunk_size=2048,\n",
    "            )\n",
    "        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)\n",
    "        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)\n",
    "        return p, s, new_accum, loss, aux_info\n",
    "\n",
    "    def distributed_apply_step(p, s, accum_grads, n_accum):\n",
    "        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)\n",
    "\n",
    "        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))\n",
    "        clip_factor = jnp.minimum(1.0, 1.0 / (global_norm + 1e-6))\n",
    "        avg_grads = jax.tree_util.tree_map(lambda g: g * clip_factor, avg_grads)\n",
    "\n",
    "        updates, new_s = tx.update(avg_grads, s, p)\n",
    "        new_p = optax.apply_updates(p, updates)\n",
    "        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)\n",
    "        return new_p, new_s, zero_accum\n",
    "\n",
    "    def distributed_val_step(p, b):\n",
    "        return compute_loss(\n",
    "            p, model_apply_wrapped, b, config,\n",
    "            rngs=None,\n",
    "            deterministic=True,\n",
    "        )\n",
    "\n",
    "    aux_info_sharding = {\n",
    "        \"ce_loss\": NamedSharding(mesh, P()),\n",
    "        \"aux_loss\": NamedSharding(mesh, P()),\n",
    "        \"z_loss\": NamedSharding(mesh, P()),\n",
    "        \"expert_utilization\": NamedSharding(mesh, P(None)),\n",
    "    }\n",
    "\n",
    "    compiled_train_micro = jax.jit(\n",
    "        distributed_train_step_micro,\n",
    "        donate_argnums=(0, 1, 4),\n",
    "        in_shardings=(\n",
    "            param_sharding,\n",
    "            opt_state_sharding,\n",
    "            {\"input_ids\": data_sharding, \"labels\": data_sharding},\n",
    "            NamedSharding(mesh, P(None)),\n",
    "            param_sharding,\n",
    "        ),\n",
    "        out_shardings=(\n",
    "            param_sharding,\n",
    "            opt_state_sharding,\n",
    "            param_sharding,\n",
    "            NamedSharding(mesh, P()),\n",
    "            aux_info_sharding,\n",
    "        ),\n",
    "    )\n",
    "\n",
    "    compiled_apply = jax.jit(\n",
    "        distributed_apply_step,\n",
    "        donate_argnums=(0, 1, 2),\n",
    "        in_shardings=(\n",
    "            param_sharding,\n",
    "            opt_state_sharding,\n",
    "            param_sharding,\n",
    "            NamedSharding(mesh, P()),\n",
    "        ),\n",
    "        out_shardings=(\n",
    "            param_sharding,\n",
    "            opt_state_sharding,\n",
    "            param_sharding,\n",
    "        ),\n",
    "    )\n",
    "\n",
    "    compiled_val = jax.jit(\n",
    "        distributed_val_step,\n",
    "        in_shardings=(param_sharding, {\"input_ids\": data_sharding, \"labels\": data_sharding}),\n",
    "        out_shardings=NamedSharding(mesh, P()),\n",
    "    )\n",
    "\n",
    "    return (compiled_train_micro, compiled_apply, compiled_val, mesh, tx, model,\n",
    "            param_sharding, opt_state_sharding, data_sharding)\n",
    "\n",
    "\n",
    "def resolve_source_files(output_dir, prefix):\n",
    "    merged_ids = os.path.join(output_dir, f\"{prefix}_input_ids.npy\")\n",
    "    merged_lbls = os.path.join(output_dir, f\"{prefix}_labels.npy\")\n",
    "    if os.path.exists(merged_ids) and os.path.exists(merged_lbls):\n",
    "        return [(merged_ids, merged_lbls)]\n",
    "\n",
    "    shard_ids_paths = sorted(\n",
    "        glob.glob(os.path.join(output_dir, f\"{prefix}_shard_ids_*.npy\")),\n",
    "        key=lambda p: int(re.search(r\"_(\\d+)\\.npy$\", p).group(1)),\n",
    "    )\n",
    "    pairs = []\n",
    "    for ids_path in shard_ids_paths:\n",
    "        lbls_path = ids_path.replace(\"_shard_ids_\", \"_shard_lbls_\")\n",
    "        if os.path.exists(lbls_path):\n",
    "            pairs.append((ids_path, lbls_path))\n",
    "    if not pairs:\n",
    "        raise FileNotFoundError(\n",
    "            f\"Не найдены файлы для prefix={prefix!r} в {output_dir} -- ни объединённого \"\n",
    "            f\"{prefix}_input_ids.npy, ни шардов {prefix}_shard_ids_*.npy. Проверьте путь.\"\n",
    "        )\n",
    "    return pairs\n",
    "\n",
    "\n",
    "def build_manifest(file_pairs):\n",
    "    manifest = []\n",
    "    total = 0\n",
    "    for ids_path, lbls_path in file_pairs:\n",
    "        n_rows = np.load(ids_path, mmap_mode=\"r\").shape[0]\n",
    "        manifest.append((ids_path, lbls_path, n_rows))\n",
    "        total += n_rows\n",
    "        print(f\"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков\")\n",
    "    print(f\"[DATA] Комбинированный пул: {total:,} блоков из {len(manifest)} файл(ов)\")\n",
    "    return manifest\n",
    "\n",
    "\n",
    "def dataloader_multi_source(file_pairs, batch_size, data_sharding, seq_len, val_split=0.05,\n",
    "                             dataset_fraction=1.0, fraction_seed=777, skip_batches=0):\n",
    "    manifest = build_manifest(file_pairs)\n",
    "    sizes = np.array([n for _, _, n in manifest])\n",
    "    offsets = np.concatenate([[0], np.cumsum(sizes)])\n",
    "    total_blocks = int(offsets[-1])\n",
    "    context_length = np.load(manifest[0][0], mmap_mode=\"r\").shape[1]\n",
    "    if context_length > seq_len:\n",
    "        context_length = seq_len\n",
    "\n",
    "    mmap_cache = {}\n",
    "\n",
    "    def _get_mmap(path):\n",
    "        arr = mmap_cache.get(path)\n",
    "        if arr is None:\n",
    "            arr = np.load(path, mmap_mode=\"r\")\n",
    "            mmap_cache[path] = arr\n",
    "        return arr\n",
    "\n",
    "    def _gather_batch(global_indices):\n",
    "        shard_of = np.searchsorted(offsets, global_indices, side=\"right\") - 1\n",
    "        ids_out = np.empty((len(global_indices), context_length), dtype=np.int32)\n",
    "        lbls_out = np.empty((len(global_indices), context_length), dtype=np.int32)\n",
    "        for s in np.unique(shard_of):\n",
    "            m = shard_of == s\n",
    "            local_idx = global_indices[m] - offsets[s]\n",
    "            ids_path, lbls_path, _ = manifest[s]\n",
    "            ids_full = _get_mmap(ids_path)[local_idx]\n",
    "            lbls_full = _get_mmap(lbls_path)[local_idx]\n",
    "            ids_out[m] = ids_full[:, :seq_len]\n",
    "            lbls_out[m] = lbls_full[:, :seq_len]\n",
    "        return ids_out, lbls_out\n",
    "\n",
    "    all_idx = np.arange(total_blocks)\n",
    "    if dataset_fraction < 1.0:\n",
    "        frac_rng = np.random.RandomState(fraction_seed)\n",
    "        n_keep = int(total_blocks * dataset_fraction)\n",
    "        all_idx = frac_rng.choice(all_idx, size=n_keep, replace=False)\n",
    "        all_idx.sort()\n",
    "        print(f\"[DATA] Подвыборка {dataset_fraction*100:.0f}%: {n_keep:,} из {total_blocks:,} блоков \"\n",
    "              f\"(seed={fraction_seed}, детерминированно между рестартами)\")\n",
    "\n",
    "    pool_size = len(all_idx)\n",
    "    val_size = int(pool_size * val_split)\n",
    "    train_size = pool_size - val_size\n",
    "\n",
    "    split_rng = np.random.RandomState(42)\n",
    "    shuffled = np.copy(all_idx)\n",
    "    split_rng.shuffle(shuffled)\n",
    "    train_idx_pool = shuffled[:train_size]\n",
    "    val_idx_pool = shuffled[train_size:]\n",
    "\n",
    "    def _generator(pool, is_train=True, skip_first=0):\n",
    "        idx_local = np.copy(pool)\n",
    "        local_rng = np.random.RandomState(123)\n",
    "        first_pass = True\n",
    "        while True:\n",
    "            if is_train:\n",
    "                local_rng.shuffle(idx_local)\n",
    "            n_steps = len(idx_local) // batch_size\n",
    "            start_step = 0\n",
    "            if first_pass and is_train:\n",
    "                start_step = skip_first % max(n_steps, 1)\n",
    "                if start_step > 0:\n",
    "                    print(f\"[DATA] Resume: пропускаем первые {start_step} микрошагов \"\n",
    "                          f\"текущего прохода датасета (уже были пройдены раньше).\")\n",
    "            first_pass = False\n",
    "            for step in range(start_step, n_steps):\n",
    "                batch_idx = idx_local[step * batch_size: (step + 1) * batch_size]\n",
    "                ids_np, lbls_np = _gather_batch(batch_idx)\n",
    "                yield {\n",
    "                    \"input_ids\": jax.device_put(jnp.array(ids_np), data_sharding),\n",
    "                    \"labels\": jax.device_put(jnp.array(lbls_np), data_sharding),\n",
    "                }\n",
    "            if not is_train:\n",
    "                break\n",
    "\n",
    "    return (\n",
    "        _generator(train_idx_pool, True, skip_first=skip_batches),\n",
    "        lambda: _generator(val_idx_pool, False),\n",
    "        train_size // batch_size,\n",
    "        val_size // batch_size,\n",
    "    )\n",
    "\n",
    "\n",
    "def main_execution():\n",
    "    ckpt_root = \"/kaggle/working/orbax_checkpoints\"\n",
    "    latest_dir = os.path.join(ckpt_root, \"latest\")\n",
    "    best_train_dir = os.path.join(ckpt_root, \"best_train\")\n",
    "    best_val_dir = os.path.join(ckpt_root, \"best_val\")\n",
    "    for d in (latest_dir, best_train_dir, best_val_dir):\n",
    "        os.makedirs(d, exist_ok=True)\n",
    "\n",
    "    mngr_latest = make_manager(latest_dir, max_to_keep=HF_LATEST_KEEP_N)\n",
    "    mngr_best_train = make_manager(best_train_dir, max_to_keep=1)\n",
    "    mngr_best_val = make_manager(best_val_dir, max_to_keep=1)\n",
    "\n",
    "    # --- Resume: local 'latest' slot first, потом HF ---\n",
    "    resume_step = mngr_latest.latest_step()\n",
    "    if resume_step is not None:\n",
    "        print(f\"[LOCAL] 📦 Found checkpoint: step {resume_step}\")\n",
    "\n",
    "    if resume_step is None and _HAS_HF:\n",
    "        resume_step = download_slot(latest_dir, \"latest\")\n",
    "        if resume_step is not None:\n",
    "            # Пересоздаём mngr_latest, чтобы он увидел скачанные шаги\n",
    "            mngr_latest = make_manager(latest_dir, max_to_keep=HF_LATEST_KEEP_N)\n",
    "\n",
    "    resume = (resume_step is not None)\n",
    "    start_epoch = 0\n",
    "    global_step = 0\n",
    "    best_val_loss = float(\"inf\")\n",
    "    best_train_loss = float(\"inf\")\n",
    "\n",
    "    config = ModelConfig(\n",
    "        d_model=768,\n",
    "        d_state=128,\n",
    "        d_conv=4,\n",
    "        expand=2,\n",
    "        n_heads=8,\n",
    "        d_latent=512,\n",
    "        d_ff=4096,\n",
    "        num_experts=8,\n",
    "        top_k=2,\n",
    "        num_layers=21,\n",
    "        layers_per_block=3,\n",
    "        vocab_size=128256,\n",
    "        dropout_rate=0.1,\n",
    "        router_aux_loss_coef=0.01,\n",
    "        router_z_loss_coef=0.0001,\n",
    "        moe_capacity_factor=1.0,\n",
    "        tie_embeddings=True,\n",
    "        label_smoothing=0.0,\n",
    "        router_noise_std=0.1,\n",
    "        use_flash_attention=True,\n",
    "        deltanet_chunk_size=256,\n",
    "    )\n",
    "    file_pairs = [\n",
    "        (\n",
    "        \"/kaggle/input/datasets/akseleu1j/codex-dataset/codex_input_ids.npy\",\n",
    "        \"/kaggle/input/datasets/akseleu1j/codex-dataset/codex_labels.npy\",\n",
    "        ),  # codex\n",
    "        (\n",
    "            \"/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_input_ids.npy\",\n",
    "            \"/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_labels.npy\",\n",
    "        ),  # kodcode\n",
    "        (\n",
    "            \"/kaggle/input/datasets/umirbayulgaisha/math-data/math_input_ids.npy\",\n",
    "            \"/kaggle/input/datasets/umirbayulgaisha/math-data/math_labels.npy\",\n",
    "        ), #math\n",
    "        (\n",
    "            \"/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_input_ids.npy\",\n",
    "            \"/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_labels.npy\",\n",
    "        ),  # rstar\n",
    "        (\n",
    "             \"/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_input_ids.npy\",\n",
    "            \"/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_labels.npy\",\n",
    "        ),  # syntheticcode\n",
    "    ]\n",
    "\n",
    "    for ids_path, lbls_path in file_pairs:\n",
    "        if not os.path.exists(ids_path):\n",
    "            raise FileNotFoundError(f\"Не найден файл: {ids_path}\")\n",
    "        if not os.path.exists(lbls_path):\n",
    "            raise FileNotFoundError(f\"Не найден файл: {lbls_path}\")\n",
    "    print(\"Все файлы найдены.\")\n",
    "\n",
    "    manifest = build_manifest(file_pairs)\n",
    "    total_blocks_full = sum(n for _, _, n in manifest)\n",
    "    total_blocks = int(total_blocks_full * DATASET_FRACTION)\n",
    "    print(f\"Всего блоков (полный пул): {total_blocks_full:,}\")\n",
    "    print(f\"Всего блоков (после {DATASET_FRACTION*100:.0f}% подвыборки): {total_blocks:,}\")\n",
    "\n",
    "    micro_batch_size = 8\n",
    "    accum_steps = 4\n",
    "    effective_batch_size = micro_batch_size * accum_steps\n",
    "    seq_len = 4096\n",
    "    epochs = 1\n",
    "    early_stop_patience = 2\n",
    "    eval_every_steps = 300\n",
    "    eval_batches = 40\n",
    "    eval_patience = 5\n",
    "\n",
    "    val_split = 0.05\n",
    "    val_size = int(total_blocks * val_split)\n",
    "    train_size = total_blocks - val_size\n",
    "\n",
    "    train_steps_per_epoch = train_size // effective_batch_size\n",
    "    total_train_steps = train_steps_per_epoch * epochs\n",
    "    micro_steps_per_epoch = train_size // micro_batch_size\n",
    "\n",
    "    print(f\"[TPU] Компиляция XLA графа под {total_train_steps} эффективных шагов \"\n",
    "          f\"({epochs} эпох(и) x {train_steps_per_epoch} шагов, accum={accum_steps})...\")\n",
    "\n",
    "    (compiled_train_micro, compiled_apply, compiled_val, mesh, tx, model,\n",
    "     param_sharding, opt_state_sharding, data_sharding) = (\n",
    "        make_shard_and_compile(config, total_train_steps, micro_batch_size, seq_len, accum_steps)\n",
    "    )\n",
    "    print(f\"[TPU] Устройств в mesh: {mesh.shape['tpu_nodes']} (FSDP: params, state и батч шардированы).\")\n",
    "\n",
    "    _sanity_stream, _, _, _ = dataloader_multi_source(\n",
    "        file_pairs, micro_batch_size, data_sharding, seq_len=seq_len,\n",
    "        dataset_fraction=DATASET_FRACTION, fraction_seed=DATASET_FRACTION_SEED,\n",
    "    )\n",
    "    print(\"[SANITY] Проверка первого батча...\")\n",
    "    test_batch = next(_sanity_stream)\n",
    "    max_label = int(jnp.max(test_batch['labels']))\n",
    "    min_label = int(jnp.min(test_batch['labels']))\n",
    "    print(f\"[SANITY] Labels range: [{min_label}, {max_label}], vocab_size={config.vocab_size}\")\n",
    "    assert max_label < config.vocab_size, f\"max_label={max_label} >= vocab_size!\"\n",
    "    valid_mask = test_batch['labels'] >= 0\n",
    "    n_valid = int(jnp.sum(valid_mask))\n",
    "    print(f\"[SANITY] Валидных labels в батче: {n_valid}/{valid_mask.size} ({100*n_valid/valid_mask.size:.1f}%)\")\n",
    "    if n_valid == 0:\n",
    "        raise ValueError(\"Все labels в первом батче маскированы (pad) — loss будет NaN!\")\n",
    "\n",
    "    ids_np_chk = jax.device_get(test_batch[\"input_ids\"])\n",
    "    lbls_np_chk = jax.device_get(test_batch[\"labels\"])\n",
    "    valid_chk = lbls_np_chk[:, :-1] != -100\n",
    "    shift_match = np.mean(lbls_np_chk[:, :-1][valid_chk] == ids_np_chk[:, 1:][valid_chk]) if valid_chk.any() else float(\"nan\")\n",
    "    same_pos_match = np.mean(lbls_np_chk == ids_np_chk)\n",
    "    print(f\"[SANITY] labels[i]==ids[i+1] (должно быть высоким): {shift_match:.2%}\")\n",
    "    print(f\"[SANITY] labels[i]==ids[i]   (должно быть низким): {same_pos_match:.2%}\")\n",
    "    if same_pos_match > 0.5:\n",
    "        raise ValueError(\n",
    "            \"labels совпадают с input_ids на тех же позициях в >50% случаев -- \"\n",
    "            \"датасет не сдвинут на 1 токен. Останавливаю обучение до фикса данных.\"\n",
    "        )\n",
    "    del _sanity_stream\n",
    "\n",
    "    global_rng = jax.random.PRNGKey(42)\n",
    "    init_params_fn = jax.jit(\n",
    "        lambda rng: model.init(rng, jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32))[\"params\"],\n",
    "        out_shardings=param_sharding,\n",
    "    )\n",
    "    params = init_params_fn(global_rng)\n",
    "    print(f\"[MEM] Доступно памяти на чипе 0: {jax.local_devices()[0].memory_stats()}\")\n",
    "    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))\n",
    "    print(f\"Общее количество параметров: {total_params:,} (≈ {total_params / 1e9:.2f} млрд)\")\n",
    "\n",
    "    weights_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))\n",
    "    n_devices_display = mesh.shape[\"tpu_nodes\"]\n",
    "    print(f\"Размер весов модели (глобально): {weights_bytes / 1e9:.2f} ГБ \"\n",
    "          f\"(с FSDP на чип реально хранится в среднем ~{weights_bytes / 1e9 / n_devices_display:.2f} ГБ)\")\n",
    "\n",
    "    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)\n",
    "\n",
    "    zero_accum = jax.jit(\n",
    "        lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),\n",
    "        out_shardings=param_sharding,\n",
    "    )(params)\n",
    "    accum_grads = zero_accum\n",
    "\n",
    "    if resume and resume_step is not None:\n",
    "        print(f\"[RESUME] ⬆️ Restoring step {resume_step} из 'latest'...\")\n",
    "        try:\n",
    "            restored = mngr_latest.restore(\n",
    "                resume_step,\n",
    "                args=ocp.args.StandardRestore({\"params\": params, \"opt_state\": opt_state}),\n",
    "            )\n",
    "            params = restored[\"params\"]\n",
    "            opt_state = restored[\"opt_state\"]\n",
    "            accum_grads = jax.jit(\n",
    "                lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),\n",
    "                out_shardings=param_sharding,\n",
    "            )(params)\n",
    "\n",
    "            meta_path = os.path.join(latest_dir, str(resume_step), \"metadata.json\")\n",
    "            if os.path.exists(meta_path):\n",
    "                with open(meta_path) as f:\n",
    "                    meta = json.load(f)\n",
    "                start_epoch = meta.get(\"epoch\", 0)\n",
    "                global_step = meta.get(\"global_step\", resume_step)\n",
    "                best_val_loss = meta.get(\"best_val_loss\", float(\"inf\"))\n",
    "                best_train_loss = meta.get(\"best_train_loss\", float(\"inf\"))\n",
    "            else:\n",
    "                global_step = resume_step\n",
    "            global_rng = jax.random.PRNGKey(42 + global_step)\n",
    "\n",
    "            # ФИКС: sanity-проверка после restore -- ловим \"застрял на ln(vocab)\"\n",
    "            # сразу, не через сотни шагов. Не идеальная защита (веса могут быть\n",
    "            # тихо повреждены не до NaN/init-уровня), но отсекает самый частый случай.\n",
    "            param_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params))))\n",
    "            has_nan = any(bool(jnp.any(jnp.isnan(x))) for x in jax.tree_util.tree_leaves(params))\n",
    "            print(f\"[RESUME DEBUG] param_norm={param_norm:.4f}, has_nan={has_nan}\")\n",
    "            if has_nan:\n",
    "                raise ValueError(\"Восстановленные params содержат NaN -- чекпоинт повреждён.\")\n",
    "\n",
    "            print(f\"[RESUME] ✅ Restored: step={global_step}, best_val={best_val_loss:.4f}, best_train={best_train_loss:.4f}\")\n",
    "        except Exception as e:\n",
    "            print(f\"[RESUME] ❌ Error: {e}. Starting fresh.\")\n",
    "            resume = False\n",
    "            global_step = 0\n",
    "    else:\n",
    "        print(\"[RESUME] 🆕 Fresh start.\")\n",
    "\n",
    "    skip_micro_steps = global_step * accum_steps\n",
    "    train_stream, val_factory, _, val_steps = dataloader_multi_source(\n",
    "        file_pairs, micro_batch_size, data_sharding, seq_len=seq_len,\n",
    "        dataset_fraction=DATASET_FRACTION, fraction_seed=DATASET_FRACTION_SEED,\n",
    "        skip_batches=skip_micro_steps,\n",
    "    )\n",
    "\n",
    "    _dummy_batch = {\n",
    "        \"input_ids\": jax.device_put(jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32), data_sharding),\n",
    "        \"labels\": jax.device_put(jnp.zeros((micro_batch_size, seq_len), dtype=jnp.int32), data_sharding),\n",
    "    }\n",
    "    _lowered = compiled_train_micro.lower(params, opt_state, _dummy_batch, global_rng, accum_grads)\n",
    "    _compiled_exec = _lowered.compile()\n",
    "    _analysis = _compiled_exec.memory_analysis()\n",
    "    print(f\"[MEM ANALYSIS] HBM temp:      {_analysis.temp_size_in_bytes / 1e9:.2f} ГБ\")\n",
    "    print(f\"[MEM ANALYSIS] HBM arguments: {_analysis.argument_size_in_bytes / 1e9:.2f} ГБ\")\n",
    "    print(f\"[MEM ANALYSIS] HBM output:    {_analysis.output_size_in_bytes / 1e9:.2f} ГБ\")\n",
    "    print(\"[TPU] Компиляция готова -- переходим к реальному обучению.\")\n",
    "\n",
    "    stopped_early = False\n",
    "    stopped_by_time_budget = False\n",
    "    eval_no_improve_count = 0\n",
    "    epochs_without_improvement = 0\n",
    "    best_eval_loss = float(\"inf\")\n",
    "    epoch = start_epoch\n",
    "\n",
    "    def _save_all_needed_slots(step, cur_train_loss_val, force_latest=True, tag=\"\"):\n",
    "        \"\"\"Сохраняет 'latest' всегда; 'best_train' -- если побит рекорд train_loss.\"\"\"\n",
    "        nonlocal best_train_loss\n",
    "        if force_latest:\n",
    "            save_slot(mngr_latest, latest_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, cur_train_loss_val)\n",
    "            upload_slot(latest_dir, \"latest\", step, tag, keep_last_n=HF_LATEST_KEEP_N)\n",
    "        if cur_train_loss_val is not None:\n",
    "            tl = float(jax.device_get(cur_train_loss_val))\n",
    "            if tl < best_train_loss:\n",
    "                best_train_loss = tl\n",
    "                save_slot(mngr_best_train, best_train_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, cur_train_loss_val)\n",
    "                upload_slot(best_train_dir, \"best_train\", step, f\"train_loss={tl:.4f}\", keep_last_n=1)\n",
    "                print(f\"[BEST_TRAIN] Новый лучший train_loss: {tl:.4f} на шаге {step}\")\n",
    "\n",
    "    def emergency_save(signum=None, frame=None):\n",
    "        print(f\"\\n🚨 [EMERGENCY] Saving step {global_step}...\")\n",
    "        try:\n",
    "            _save_all_needed_slots(global_step, None, force_latest=True, tag=\"EMERGENCY\")\n",
    "            print(f\"🚨 ✅ Emergency save done (local + HF): step {global_step}\")\n",
    "        except Exception as e:\n",
    "            print(f\"🚨 ❌ Emergency save failed: {e}\")\n",
    "        sys.exit(0)\n",
    "\n",
    "    signal.signal(signal.SIGTERM, emergency_save)\n",
    "    signal.signal(signal.SIGINT, emergency_save)\n",
    "\n",
    "    total_tokens_processed = 0\n",
    "    epoch_start_time = time.perf_counter()\n",
    "    last_ckpt_time = time.perf_counter()\n",
    "    session_start_time = time.perf_counter()\n",
    "\n",
    "    for epoch in range(start_epoch, epochs):\n",
    "        for micro_step in range(micro_steps_per_epoch):\n",
    "            global_rng, step_rng = jax.random.split(global_rng)\n",
    "\n",
    "            _t0 = time.perf_counter()\n",
    "            try:\n",
    "                batch = next(train_stream)\n",
    "            except StopIteration:\n",
    "                print(\"[DATA] Поток данных исчерпан для этой эпохи.\")\n",
    "                break\n",
    "            _t_data = time.perf_counter() - _t0\n",
    "\n",
    "            total_tokens_processed += micro_batch_size * seq_len\n",
    "\n",
    "            _t1 = time.perf_counter()\n",
    "            params, opt_state, accum_grads, train_loss, aux_info = compiled_train_micro(\n",
    "                params, opt_state, batch, step_rng, accum_grads\n",
    "            )\n",
    "            if micro_step < 30:\n",
    "                jax.block_until_ready(train_loss)\n",
    "            _t_compute = time.perf_counter() - _t1\n",
    "\n",
    "            if (micro_step + 1) % accum_steps == 0:\n",
    "                effective_step = (micro_step + 1) // accum_steps\n",
    "\n",
    "                _t_apply = time.perf_counter()\n",
    "                params, opt_state, accum_grads = compiled_apply(\n",
    "                    params, opt_state, accum_grads, accum_steps\n",
    "                )\n",
    "                if micro_step < 30:\n",
    "                    jax.block_until_ready(params)\n",
    "                _t_apply_total = time.perf_counter() - _t_apply\n",
    "\n",
    "                global_step += 1\n",
    "\n",
    "                now = time.perf_counter()\n",
    "                if now - last_ckpt_time >= CHECKPOINT_EVERY_SECONDS:\n",
    "                    print(f\"[CKPT] 💾 Цикл сохранения на шаге {global_step} (прошло {(now - last_ckpt_time)/60:.1f} мин)...\")\n",
    "                    _save_all_needed_slots(global_step, train_loss, force_latest=True)\n",
    "                    last_ckpt_time = time.perf_counter()  # ФИКС: после реальной длительности сейва, не до\n",
    "\n",
    "                elapsed_session = time.perf_counter() - session_start_time\n",
    "                if elapsed_session >= SESSION_TIME_BUDGET_SECONDS:\n",
    "                    print(f\"[SESSION LIMIT] Достигнут бюджет времени сессии \"\n",
    "                          f\"({elapsed_session/3600:.2f} ч) -- сохраняюсь и завершаюсь gracefully...\")\n",
    "                    _save_all_needed_slots(global_step, train_loss, force_latest=True, tag=\"SESSION_LIMIT\")\n",
    "                    stopped_by_time_budget = True\n",
    "                    stopped_early = True\n",
    "                    break\n",
    "\n",
    "                if micro_step < 30:\n",
    "                    total_step_time = _t_compute + _t_apply_total\n",
    "                    print(f\"[TIMING] effective step {effective_step}: \"\n",
    "                          f\"данные={_t_data*1000:.0f}мс  \"\n",
    "                          f\"TPU compute={_t_compute*1000:.0f}мс  \"\n",
    "                          f\"apply={_t_apply_total*1000:.0f}мс  \"\n",
    "                          f\"(доля данных: {_t_data/(total_step_time+_t_data)*100:.0f}%)\")\n",
    "\n",
    "                if effective_step % 10 == 0:\n",
    "                    print(\n",
    "                        f\"Epoch: {epoch} | Step: {effective_step}/{train_steps_per_epoch} | \"\n",
    "                        f\"Global Step: {global_step} | Train Loss: {jax.device_get(train_loss):.4f} \"\n",
    "                        f\"(ce={jax.device_get(aux_info['ce_loss']):.4f} \"\n",
    "                        f\"aux={jax.device_get(aux_info['aux_loss']):.4f} \"\n",
    "                        f\"z={jax.device_get(aux_info['z_loss']):.5f}) | \"\n",
    "                        f\"best_train={best_train_loss:.4f}\"\n",
    "                    )\n",
    "                    if aux_info[\"expert_utilization\"] is not None:\n",
    "                        util = jax.device_get(aux_info[\"expert_utilization\"])\n",
    "                        util_std_per_layer = util.std(axis=-1)\n",
    "                        worst_layer = int(util_std_per_layer.argmax())\n",
    "                        print(\n",
    "                            f\"           expert utilization std (max over layers, layer {worst_layer}): \"\n",
    "                            f\"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts}\"\n",
    "                        )\n",
    "\n",
    "                if global_step % eval_every_steps == 0:\n",
    "                    val_stream = val_factory()\n",
    "                    eval_loss = 0.0\n",
    "                    n_batches_done = 0\n",
    "                    for _ in range(eval_batches):\n",
    "                        try:\n",
    "                            eval_batch = next(val_stream)\n",
    "                        except StopIteration:\n",
    "                            break\n",
    "                        eval_loss += jax.device_get(compiled_val(params, eval_batch))\n",
    "                        n_batches_done += 1\n",
    "                    eval_loss /= max(n_batches_done, 1)\n",
    "                    print(f\"[EVAL] Step {global_step}: val loss (частичный, {n_batches_done} батчей) = {eval_loss:.4f}\")\n",
    "\n",
    "                    if eval_loss < best_eval_loss:\n",
    "                        best_eval_loss = eval_loss\n",
    "                        eval_no_improve_count = 0\n",
    "                        if eval_loss < best_val_loss:\n",
    "                            best_val_loss = eval_loss\n",
    "                            save_slot(mngr_best_val, best_val_dir, global_step, params, opt_state, epoch, best_val_loss, best_train_loss)\n",
    "                            upload_slot(best_val_dir, \"best_val\", global_step, f\"val_loss={eval_loss:.4f}\", keep_last_n=1)\n",
    "                            print(f\"[BEST_VAL] Новый лучший val_loss: {best_val_loss:.4f} на шаге {global_step}\")\n",
    "                    else:\n",
    "                        eval_no_improve_count += 1\n",
    "                        if eval_no_improve_count >= eval_patience:\n",
    "                            print(\n",
    "                                f\"[EARLY STOP] Частичный val loss не улучшался {eval_patience} \"\n",
    "                                \"проверок подряд. Останавливаю обучение немедленно.\"\n",
    "                            )\n",
    "                            _save_all_needed_slots(global_step, train_loss, force_latest=True, tag=\"EARLY_STOP\")\n",
    "                            print(f\"[ORBAX] Финальные чекпоинты (шаг {global_step}) сохранены.\")\n",
    "                            stopped_early = True\n",
    "                            break\n",
    "\n",
    "            else:\n",
    "                if micro_step < 30:\n",
    "                    print(f\"[TIMING] micro step {micro_step}: \"\n",
    "                          f\"данные={_t_data*1000:.0f}мс  \"\n",
    "                          f\"TPU={_t_compute*1000:.0f}мс  (accumulating)\")\n",
    "\n",
    "        if stopped_early:\n",
    "            break\n",
    "\n",
    "        print(f\"--- Эпоха {epoch} завершена. Запуск распределенной кросс-валидации ---\")\n",
    "        val_stream = val_factory()\n",
    "        total_val_loss = 0.0\n",
    "        for _ in range(val_steps):\n",
    "            total_val_loss += jax.device_get(compiled_val(params, next(val_stream)))\n",
    "\n",
    "        mean_val_loss = total_val_loss / val_steps\n",
    "        print(f\"===> Эпоха: {epoch} | ИТОГОВЫЙ СРЕДНИЙ VALIDATION LOSS: {mean_val_loss:.4f} <===\")\n",
    "\n",
    "        epoch_elapsed = time.perf_counter() - epoch_start_time\n",
    "        tokens_per_sec = total_tokens_processed / epoch_elapsed\n",
    "        print(f\"Средняя скорость эпохи: {tokens_per_sec / 1e6:.2f} млн токенов/сек\")\n",
    "\n",
    "        total_tokens_processed = 0\n",
    "        epoch_start_time = time.perf_counter()\n",
    "\n",
    "        _save_all_needed_slots(global_step, None, force_latest=True, tag=\"EPOCH_END\")\n",
    "\n",
    "        if mean_val_loss < best_val_loss:\n",
    "            best_val_loss = mean_val_loss\n",
    "            epochs_without_improvement = 0\n",
    "            save_slot(mngr_best_val, best_val_dir, global_step, params, opt_state, epoch, best_val_loss, best_train_loss)\n",
    "            upload_slot(best_val_dir, \"best_val\", global_step, f\"val_loss={mean_val_loss:.4f} EPOCH_END\", keep_last_n=1)\n",
    "            print(f\"[BEST_VAL] Новый лучший val_loss ({best_val_loss:.4f}) -- сохранён\")\n",
    "        else:\n",
    "            epochs_without_improvement += 1\n",
    "            print(\n",
    "                f\"[EARLY STOP] val loss не улучшился {epochs_without_improvement} эпох(и) подряд \"\n",
    "                f\"(лучший: {best_val_loss:.4f})\"\n",
    "            )\n",
    "            if epochs_without_improvement >= early_stop_patience:\n",
    "                print(\n",
    "                    f\"[EARLY STOP] Останавливаю обучение -- val loss не улучшался \"\n",
    "                    f\"{early_stop_patience} эпохи подряд.\"\n",
    "                )\n",
    "                break\n",
    "\n",
    "    if stopped_by_time_budget:\n",
    "        print(f\"[SESSION LIMIT] Обучение остановлено по бюджету времени сессии на шаге {global_step}. \"\n",
    "              f\"Запустите скрипт заново для продолжения.\")\n",
    "    print(\"Обучение завершено (для этой сессии).\")\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main_execution()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a72f8da1",
   "metadata": {
    "execution": {
     "execution_failed": "2026-08-08T19:30:35.028Z"
    },
    "papermill": {
     "duration": 0.016947,
     "end_time": "2026-08-09T04:00:49.895522+00:00",
     "exception": false,
     "start_time": "2026-08-09T04:00:49.878575+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e5fc447d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-09T04:00:49.932070Z",
     "iopub.status.busy": "2026-08-09T04:00:49.931861Z",
     "iopub.status.idle": "2026-08-09T04:00:50.042853Z",
     "shell.execute_reply": "2026-08-09T04:00:50.041562Z"
    },
    "papermill": {
     "duration": 0.131624,
     "end_time": "2026-08-09T04:00:50.043672+00:00",
     "exception": false,
     "start_time": "2026-08-09T04:00:49.912048+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "file_pairs = [\n",
      "    (\n",
      "        \"/kaggle/input/datasets/akseleu1j/codex-dataset/codex_input_ids.npy\",\n",
      "        \"/kaggle/input/datasets/akseleu1j/codex-dataset/codex_labels.npy\",\n",
      "    ),\n",
      "    (\n",
      "        \"/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_input_ids.npy\",\n",
      "        \"/kaggle/input/datasets/akseleu1j/kodcode-dataset/kodcode_labels.npy\",\n",
      "    ),\n",
      "    (\n",
      "        \"/kaggle/input/datasets/umirbayulgaisha/math-data/math_input_ids.npy\",\n",
      "        \"/kaggle/input/datasets/umirbayulgaisha/math-data/math_labels.npy\",\n",
      "    ),\n",
      "    (\n",
      "        \"/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_input_ids.npy\",\n",
      "        \"/kaggle/input/datasets/akseleu1j/rstar-dataset/rstar_labels.npy\",\n",
      "    ),\n",
      "    (\n",
      "        \"/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_input_ids.npy\",\n",
      "        \"/kaggle/input/datasets/akseleu1j/sytetic-dataset/syntheticcode_labels.npy\",\n",
      "    ),\n",
      "]\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import glob\n",
    "\n",
    "def find_dataset_pairs(\n",
    "    base_path=\"/kaggle/input\",\n",
    "    prefixes=None,  # если None, то автоматически определяются\n",
    "):\n",
    "    \"\"\"\n",
    "    Находит все пары input_ids / labels .npy файлов в /kaggle/input.\n",
    "    Возвращает список кортежей (ids_path, lbls_path).\n",
    "    \"\"\"\n",
    "    if prefixes is None:\n",
    "        # Автоматически определяем префиксы по именам файлов\n",
    "        ids_files = glob.glob(os.path.join(base_path, \"**\", \"*_input_ids.npy\"), recursive=True)\n",
    "        prefixes = sorted({os.path.basename(f).replace(\"_input_ids.npy\", \"\") for f in ids_files})\n",
    "    \n",
    "    pairs = []\n",
    "    for prefix in prefixes:\n",
    "        # Ищем файлы по шаблону\n",
    "        pattern = os.path.join(base_path, \"**\", f\"{prefix}_input_ids.npy\")\n",
    "        ids_files = glob.glob(pattern, recursive=True)\n",
    "        for ids_path in ids_files:\n",
    "            lbls_path = ids_path.replace(\"_input_ids.npy\", \"_labels.npy\")\n",
    "            if os.path.exists(lbls_path):\n",
    "                pairs.append((ids_path, lbls_path))\n",
    "    return pairs\n",
    "\n",
    "# Находим все пары\n",
    "file_pairs = find_dataset_pairs()\n",
    "\n",
    "# Выводим в формате, удобном для копирования в train.py\n",
    "print(\"file_pairs = [\")\n",
    "for ids_path, lbls_path in file_pairs:\n",
    "    print(f\"    (\\n        \\\"{ids_path}\\\",\\n        \\\"{lbls_path}\\\",\\n    ),\")\n",
    "print(\"]\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4e71839e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-09T04:00:50.079620Z",
     "iopub.status.busy": "2026-08-09T04:00:50.079358Z",
     "iopub.status.idle": "2026-08-09T04:00:50.084037Z",
     "shell.execute_reply": "2026-08-09T04:00:50.083033Z"
    },
    "papermill": {
     "duration": 0.023775,
     "end_time": "2026-08-09T04:00:50.084717+00:00",
     "exception": false,
     "start_time": "2026-08-09T04:00:50.060942+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "    config = ModelConfig(\n",
    "        d_state=128,\n",
    "        d_conv=4,\n",
    "        expand=2,\n",
    "        n_heads=8,            # d_head=96, ок для RoPE\n",
    "        d_latent=512,         # было 256\n",
    "        d_ff=6144,            # было 3072\n",
    "        num_experts=8,\n",
    "        top_k=2,\n",
    "        num_layers=21,\n",
    "        layers_per_block=3,\n",
    "        vocab_size=151936,\n",
    "        dropout_rate=0.1,\n",
    "        router_aux_loss_coef=0.01,\n",
    "        router_z_loss_coef=0.0001,\n",
    "        moe_capacity_factor=1.25,\n",
    "        tie_embeddings=True,\n",
    "        label_smoothing=0.05,\n",
    "        router_noise_std=0.3,\n",
    "        use_flash_attention=True,\n",
    "        deltanet_chunk_size=256,\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8fa947da",
   "metadata": {
    "papermill": {
     "duration": 0.015668,
     "end_time": "2026-08-09T04:00:50.117780+00:00",
     "exception": false,
     "start_time": "2026-08-09T04:00:50.102112+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 30192.04362,
   "end_time": "2026-08-09T04:01:00.254902+00:00",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-08-08T19:37:48.211282+00:00",
   "version": "2.7.0"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {
     "0abba2b23fe04cc38c28caed30b52565": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_55e5d8b0b5de41e8b53d5c2d669a2c29",
       "placeholder": "​",
       "style": "IPY_MODEL_f5ed5beeb1e34a67b37afab68093f6eb",
       "tabbable": null,
       "tooltip": null,
       "value": " 16/16 [00:12&lt;00:00,  1.55s/it]"
      }
     },
     "1783e52d411b4ff090bbd435a34397e6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_8e2de34bc47e48c888b210343b7e227b",
       "max": 16.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_99790daf9b00400ea937894dc78eb149",
       "tabbable": null,
       "tooltip": null,
       "value": 16.0
      }
     },
     "1a443714db704a99b2cbd49a439a1726": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_ab8245a767d94aeb8c4d0c7dbaf08c03",
       "placeholder": "​",
       "style": "IPY_MODEL_e35883cee9984621be1147933f3db867",
       "tabbable": null,
       "tooltip": null,
       "value": "Download complete: 100%"
      }
     },
     "21cceb54b3ab40248ddf6cb9e7e7b9fc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     },
     "313a973bdf0e49d8ac054e791398d7a1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "3346ba1d9c1c4b7cb7ecedbabac59717": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     },
     "4117392a080d452f8899a46b1a24de4f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": "20px"
      }
     },
     "55e5d8b0b5de41e8b53d5c2d669a2c29": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "8e2de34bc47e48c888b210343b7e227b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "922d8ca2726a473a85955edecd95b464": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_1a443714db704a99b2cbd49a439a1726",
        "IPY_MODEL_b9b2be2c15b6453e8d83a0b4853ef49d",
        "IPY_MODEL_e24a68f1503f4c07bbbbcfc5decd0bd1"
       ],
       "layout": "IPY_MODEL_b627da0080034eabaa100fde75bde3e8",
       "tabbable": null,
       "tooltip": null
      }
     },
     "92c375506a9a4219a00ba7bedf4dd824": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_eb2502b0abb948f8b7b02b1b28dbc198",
       "placeholder": "​",
       "style": "IPY_MODEL_21cceb54b3ab40248ddf6cb9e7e7b9fc",
       "tabbable": null,
       "tooltip": null,
       "value": "Fetching 16 files: 100%"
      }
     },
     "99790daf9b00400ea937894dc78eb149": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "9a61b615f2fa4d0ca387d3a3aa49705d": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "ab8245a767d94aeb8c4d0c7dbaf08c03": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "ad78cd43adca4d0aa9662fb7c074d3a3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "b627da0080034eabaa100fde75bde3e8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "b9b2be2c15b6453e8d83a0b4853ef49d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_4117392a080d452f8899a46b1a24de4f",
       "max": 1.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_ad78cd43adca4d0aa9662fb7c074d3a3",
       "tabbable": null,
       "tooltip": null,
       "value": 1.0
      }
     },
     "d9c8b5559bae4940b47dec1c00315e54": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_92c375506a9a4219a00ba7bedf4dd824",
        "IPY_MODEL_1783e52d411b4ff090bbd435a34397e6",
        "IPY_MODEL_0abba2b23fe04cc38c28caed30b52565"
       ],
       "layout": "IPY_MODEL_9a61b615f2fa4d0ca387d3a3aa49705d",
       "tabbable": null,
       "tooltip": null
      }
     },
     "e24a68f1503f4c07bbbbcfc5decd0bd1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_313a973bdf0e49d8ac054e791398d7a1",
       "placeholder": "​",
       "style": "IPY_MODEL_3346ba1d9c1c4b7cb7ecedbabac59717",
       "tabbable": null,
       "tooltip": null,
       "value": " 2.94G/2.94G [00:13&lt;00:00, 229MB/s]"
      }
     },
     "e35883cee9984621be1147933f3db867": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     },
     "eb2502b0abb948f8b7b02b1b28dbc198": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "f5ed5beeb1e34a67b37afab68093f6eb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     }
    },
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
