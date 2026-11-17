# Image retrieval: conv_test

Input `conv_test.json`, 256 px center crop, global pooling, best checkpoint, same-slide images kept.
Queries / images: composition 1266 / 1453, texture 1093 / 1483, porosity 1173 / 1506, diagenesis 903 / 1505.
Per-attribute columns are mAP; the last five are the mean over the four attributes. Sorted by mean mAP. Similarity `q2q` = best query-to-query match (Q-Former max pooling).

| # | encoder | sim | comp. mAP | texture mAP | porosity mAP | diag. mAP | R@1 | R@5 | R@10 | mAP | nDCG@10 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GEO-DINO2048-192-t50-r32-a64-lr1-composition-augsoft-long | cos | 0.244 | 0.319 | 0.315 | 0.514 | 0.487 | 0.663 | 0.731 | 0.348 | 0.384 |
| 2 | GEO-DINO2048-192-t40-r32-a64-lr1-composition-augsoft-long | cos | 0.238 | 0.311 | 0.312 | 0.504 | 0.479 | 0.651 | 0.717 | 0.341 | 0.379 |
| 3 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-augsoft-long | cos | 0.234 | 0.305 | 0.298 | 0.488 | 0.462 | 0.638 | 0.711 | 0.331 | 0.373 |
| 4 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-aug-long | cos | 0.231 | 0.300 | 0.295 | 0.479 | 0.455 | 0.639 | 0.711 | 0.326 | 0.373 |
| 5 | GEO-DINO2048-192-t30-r32-a64-lr1-composition-augsoft-long | cos | 0.228 | 0.299 | 0.296 | 0.480 | 0.451 | 0.626 | 0.690 | 0.326 | 0.370 |
| 6 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-geometric-long | cos | 0.240 | 0.296 | 0.299 | 0.465 | 0.460 | 0.637 | 0.700 | 0.325 | 0.385 |
| 7 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-augsoft-cos-long | cos | 0.227 | 0.299 | 0.293 | 0.475 | 0.449 | 0.625 | 0.703 | 0.324 | 0.369 |
| 8 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-aug | cos | 0.232 | 0.293 | 0.286 | 0.455 | 0.441 | 0.624 | 0.698 | 0.316 | 0.370 |
| 9 | GEOv4-DINO2048-192-t80-r32-a64-lr1-composition-aug-cos | cos | 0.229 | 0.290 | 0.283 | 0.456 | 0.434 | 0.619 | 0.698 | 0.315 | 0.371 |
| 10 | FOSSIL512-E5-linear+both16-composition-aug-long | cos | 0.230 | 0.280 | 0.288 | 0.457 | 0.455 | 0.629 | 0.692 | 0.314 | 0.371 |
| 11 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-aug-cos2-long | cos | 0.222 | 0.288 | 0.282 | 0.452 | 0.438 | 0.612 | 0.684 | 0.311 | 0.366 |
| 12 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-aug-cos | cos | 0.235 | 0.284 | 0.283 | 0.442 | 0.432 | 0.598 | 0.668 | 0.311 | 0.368 |
| 13 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-aug-cos-long | cos | 0.215 | 0.286 | 0.281 | 0.448 | 0.429 | 0.598 | 0.684 | 0.308 | 0.361 |
| 14 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition-shift0.2 | cos | 0.230 | 0.276 | 0.269 | 0.421 | 0.410 | 0.612 | 0.684 | 0.299 | 0.376 |
| 15 | FOSSIL512-E5-linear+both16-composition-long | cos | 0.229 | 0.261 | 0.270 | 0.423 | 0.423 | 0.601 | 0.672 | 0.296 | 0.377 |
| 16 | GEOv4-DINO1024-192-t100-r32-a64-lr4-composition | cos | 0.217 | 0.264 | 0.251 | 0.398 | 0.386 | 0.579 | 0.668 | 0.283 | 0.360 |
| 17 | FOSSIL512-E5-linear+both16-composition-geometric-long | cos | 0.216 | 0.248 | 0.260 | 0.396 | 0.403 | 0.569 | 0.638 | 0.280 | 0.364 |
| 18 | FOSSIL512-E5-linear+both4-composition-long | cos | 0.214 | 0.242 | 0.248 | 0.392 | 0.396 | 0.559 | 0.626 | 0.274 | 0.365 |
| 19 | GEO-DINO2048-192-t20-r32-a64-lr1-composition-augsoft-long | cos | 0.201 | 0.244 | 0.247 | 0.394 | 0.376 | 0.555 | 0.635 | 0.272 | 0.340 |
| 20 | GEOv4-DINO1024-192-t100-r12-a24-lr4-composition | cos | 0.202 | 0.242 | 0.236 | 0.374 | 0.354 | 0.551 | 0.624 | 0.264 | 0.351 |
| 21 | GEOv4-DINO-192-t100-r32-a64-composition | cos | 0.209 | 0.240 | 0.233 | 0.363 | 0.356 | 0.536 | 0.627 | 0.261 | 0.353 |
| 22 | GEOv4-DINO1024-192-t30-r32-a64-lr4-composition | cos | 0.202 | 0.232 | 0.239 | 0.363 | 0.353 | 0.544 | 0.633 | 0.259 | 0.340 |
| 23 | GEOv4-DINO1024-192-t100-r32-a64-lr1-composition | cos | 0.208 | 0.234 | 0.233 | 0.359 | 0.355 | 0.545 | 0.625 | 0.259 | 0.349 |
| 24 | GEOv4-DINO1024-192-t100-r32-a64-lr2-composition | cos | 0.201 | 0.240 | 0.232 | 0.358 | 0.342 | 0.536 | 0.626 | 0.258 | 0.350 |
| 25 | GEOv4-DINO2048-192-t100-r12-a24-lr1-composition | cos | 0.202 | 0.227 | 0.235 | 0.356 | 0.346 | 0.536 | 0.620 | 0.255 | 0.346 |
| 26 | GEOv4-DINO-192-t100-r32-a64-composition-MP | cos | 0.201 | 0.237 | 0.229 | 0.351 | 0.354 | 0.531 | 0.619 | 0.255 | 0.351 |
| 27 | GEOv4-DINO-48-t100-r12-a24-composition | cos | 0.202 | 0.231 | 0.230 | 0.355 | 0.347 | 0.538 | 0.627 | 0.254 | 0.351 |
| 28 | GEOv4-DINO-48-t100-r32-a64-composition-MP | cos | 0.208 | 0.232 | 0.227 | 0.348 | 0.340 | 0.538 | 0.618 | 0.254 | 0.349 |
| 29 | GEOv4-DINO-48-t100-r4-a8-composition | cos | 0.205 | 0.236 | 0.228 | 0.343 | 0.329 | 0.536 | 0.620 | 0.253 | 0.350 |
| 30 | GEOv4-DINO1024-192-t100-r12-a24-lr2-composition | cos | 0.201 | 0.233 | 0.232 | 0.346 | 0.336 | 0.537 | 0.628 | 0.253 | 0.351 |
| 31 | GEOv4-DINO2048-192-t100-r32-a64-lr1-composition | cos | 0.196 | 0.234 | 0.228 | 0.349 | 0.343 | 0.527 | 0.606 | 0.252 | 0.343 |
| 32 | GEOv4-DINO2048-192-t100-r16-a32-lr1-composition-long | cos | 0.201 | 0.230 | 0.227 | 0.348 | 0.335 | 0.539 | 0.613 | 0.251 | 0.345 |
| 33 | GEOv4-DINO-48-t30-r4-a8-composition | cos | 0.196 | 0.228 | 0.227 | 0.343 | 0.324 | 0.514 | 0.608 | 0.248 | 0.340 |
| 34 | GEOv4-DINO1024-192-t30-r12-a24-lr1-composition | cos | 0.199 | 0.223 | 0.228 | 0.340 | 0.319 | 0.537 | 0.630 | 0.247 | 0.344 |
| 35 | GEOv4-DINO-192-t100-r12-a24-composition | cos | 0.202 | 0.220 | 0.223 | 0.343 | 0.319 | 0.535 | 0.614 | 0.247 | 0.345 |
| 36 | GEOv4-DINO1024-192-t100-r12-a24-lr1-composition | cos | 0.204 | 0.223 | 0.222 | 0.340 | 0.327 | 0.529 | 0.617 | 0.247 | 0.345 |
| 37 | GEOv4-DINO-48-t100-r12-a24-composition-MP | cos | 0.201 | 0.227 | 0.225 | 0.336 | 0.337 | 0.513 | 0.602 | 0.247 | 0.349 |
| 38 | GEOv4-DINO-192-t30-r12-a24-composition | cos | 0.198 | 0.222 | 0.228 | 0.338 | 0.327 | 0.527 | 0.620 | 0.247 | 0.345 |
| 39 | GEOv4-DINO-192-t100-r4-a8-composition | cos | 0.200 | 0.225 | 0.224 | 0.337 | 0.339 | 0.530 | 0.621 | 0.247 | 0.349 |
| 40 | GEOv4-DINO-48-t30-r12-a24-composition | cos | 0.195 | 0.222 | 0.226 | 0.335 | 0.338 | 0.513 | 0.606 | 0.244 | 0.341 |
| 41 | GEOv4-DINO-48-t100-r32-a64-composition | cos | 0.201 | 0.224 | 0.219 | 0.333 | 0.333 | 0.533 | 0.612 | 0.244 | 0.343 |
| 42 | GEOv4-DINO1024-192-t30-r32-a64-lr2-composition | cos | 0.188 | 0.224 | 0.222 | 0.336 | 0.320 | 0.525 | 0.612 | 0.242 | 0.333 |
| 43 | GEOv4-DINO-48-t30-r32-a64-composition | cos | 0.194 | 0.221 | 0.222 | 0.331 | 0.328 | 0.517 | 0.617 | 0.242 | 0.339 |
| 44 | GEOv4-DINO2048-192-t30-r12-a24-lr1-composition | cos | 0.194 | 0.221 | 0.218 | 0.323 | 0.313 | 0.512 | 0.605 | 0.239 | 0.335 |
| 45 | GEOv4-DINO1024-192-t30-r12-a24-lr4-composition | cos | 0.187 | 0.217 | 0.218 | 0.330 | 0.316 | 0.527 | 0.612 | 0.238 | 0.335 |
| 46 | FOSSIL512-E5-linear+vision16-textnoproj-composition-geometric-long | cos | 0.185 | 0.204 | 0.216 | 0.346 | 0.358 | 0.504 | 0.574 | 0.238 | 0.324 |
| 47 | GEOv4-DINO1024-192-t30-r12-a24-lr2-composition | cos | 0.191 | 0.217 | 0.214 | 0.322 | 0.321 | 0.508 | 0.600 | 0.236 | 0.336 |
| 48 | GEOv4-DINO2048-192-t30-r32-a64-lr1-composition | cos | 0.184 | 0.213 | 0.215 | 0.318 | 0.307 | 0.506 | 0.596 | 0.233 | 0.334 |
| 49 | GEOv4-DINO1024-192-t30-r32-a64-lr1-composition | cos | 0.186 | 0.215 | 0.212 | 0.316 | 0.300 | 0.515 | 0.607 | 0.232 | 0.335 |
| 50 | GEOv4-DINO-192-t100-r32-a64-composition-vh | cos | 0.175 | 0.195 | 0.196 | 0.282 | 0.291 | 0.469 | 0.563 | 0.212 | 0.322 |
| 51 | fossil | cos | 0.165 | 0.189 | 0.196 | 0.297 | 0.302 | 0.480 | 0.556 | 0.212 | 0.312 |
| 52 | FOSSIL512-E5-qformer6-t30-textnoproj-composition-long | q2q | 0.165 | 0.183 | 0.194 | 0.294 | 0.295 | 0.455 | 0.533 | 0.209 | 0.298 |
| 53 | GEOv4-DINO-192-t100-composition-vh | cos | 0.173 | 0.188 | 0.189 | 0.272 | 0.294 | 0.452 | 0.531 | 0.205 | 0.323 |
| 54 | GEO-DINO2048-192-t100-r32-a64-lr1-composition-long | cos | 0.154 | 0.191 | 0.186 | 0.277 | 0.275 | 0.452 | 0.531 | 0.202 | 0.303 |
| 55 | GEOv4-DINO-48-t100-composition-vh | cos | 0.157 | 0.168 | 0.166 | 0.244 | 0.261 | 0.417 | 0.503 | 0.184 | 0.306 |
| 56 | GEOv4-DINO-48-t100-original_recipe-composition | cos | 0.151 | 0.163 | 0.164 | 0.231 | 0.252 | 0.402 | 0.481 | 0.177 | 0.302 |
| 57 | GEOv4-DINO-48-t30-original_recipe-composition | cos | 0.151 | 0.163 | 0.164 | 0.231 | 0.252 | 0.402 | 0.481 | 0.177 | 0.302 |
| 58 | dinov3 | cos | 0.145 | 0.159 | 0.158 | 0.240 | 0.258 | 0.407 | 0.490 | 0.175 | 0.287 |
| 59 | dinotxt | cos | 0.150 | 0.161 | 0.160 | 0.227 | 0.250 | 0.401 | 0.480 | 0.174 | 0.299 |
| 60 | GEOv4-DINO-48-t100-r32-a64-composition-vh | cos | 0.149 | 0.156 | 0.151 | 0.225 | 0.240 | 0.396 | 0.472 | 0.170 | 0.293 |
| 61 | FOSSIL512-E5-linear+both16-sig-composition-long | cos | 0.120 | 0.133 | 0.138 | 0.211 | 0.198 | 0.355 | 0.436 | 0.150 | 0.242 |
| 62 | FOSSIL512-E5-qformer+text16-composition-long | q2q | 0.125 | 0.131 | 0.140 | 0.205 | 0.189 | 0.351 | 0.444 | 0.150 | 0.252 |
| 63 | FOSSIL512-E5-qformer-384-composition-long | q2q | 0.125 | 0.137 | 0.136 | 0.200 | 0.190 | 0.329 | 0.408 | 0.149 | 0.248 |
| 64 | FOSSIL512-E5-qformer-t30-composition-long | q2q | 0.121 | 0.134 | 0.136 | 0.206 | 0.211 | 0.354 | 0.423 | 0.149 | 0.250 |
| 65 | DINOTXT-E5-VH-384-composition-long | cos | 0.127 | 0.139 | 0.136 | 0.194 | 0.206 | 0.350 | 0.429 | 0.149 | 0.264 |
| 66 | FOSSIL512-E5-qformer+textnoproj-composition-long | q2q | 0.119 | 0.134 | 0.128 | 0.199 | 0.196 | 0.342 | 0.424 | 0.145 | 0.253 |
| 67 | FOSSIL512-E5-qformer-t30-textnoproj-composition-long | q2q | 0.116 | 0.117 | 0.127 | 0.193 | 0.198 | 0.319 | 0.385 | 0.138 | 0.240 |
| 68 | FOSSIL512-E5-qformer6layer-composition-long | q2q | 0.114 | 0.128 | 0.118 | 0.175 | 0.169 | 0.306 | 0.394 | 0.134 | 0.240 |
| 69 | FOSSIL512-E5-qformer-t30-2lr-composition-long | q2q | 0.110 | 0.121 | 0.118 | 0.176 | 0.184 | 0.323 | 0.397 | 0.131 | 0.239 |
| 70 | FOSSIL512-E5-qformer12layer-composition-long | q2q | 0.118 | 0.120 | 0.117 | 0.168 | 0.164 | 0.293 | 0.385 | 0.131 | 0.237 |
| 71 | FOSSIL512-E5-qformer-cooling1-composition-long | q2q | 0.111 | 0.118 | 0.115 | 0.170 | 0.175 | 0.291 | 0.355 | 0.128 | 0.232 |
| 72 | FOSSIL512-E5-64qformer-t30-composition-long | q2q | 0.112 | 0.109 | 0.116 | 0.175 | 0.157 | 0.286 | 0.367 | 0.128 | 0.228 |
| 73 | FOSSIL512-E5-qformermean-t30-textnoproj-composition-long | cos | 0.105 | 0.108 | 0.111 | 0.161 | 0.153 | 0.263 | 0.326 | 0.121 | 0.217 |
| 74 | DINOTXT-E5-VH-frozentext-composition-long | cos | 0.110 | 0.103 | 0.106 | 0.148 | 0.164 | 0.277 | 0.334 | 0.117 | 0.229 |
| 75 | FOSSIL512-E5-qformermean-t30-composition-long | cos | 0.095 | 0.105 | 0.108 | 0.155 | 0.145 | 0.276 | 0.360 | 0.116 | 0.214 |
| 76 | FOSSIL512-E5-qformer-t30-cosine10-composition-long | q2q | 0.106 | 0.103 | 0.098 | 0.147 | 0.150 | 0.254 | 0.317 | 0.114 | 0.223 |
| 77 | FOSSIL512-E5-qformer-composition-long | q2q | 0.087 | 0.101 | 0.096 | 0.139 | 0.142 | 0.239 | 0.307 | 0.106 | 0.206 |
| 78 | FOSSIL512-E5-qformer-MP-composition-long | q2q | 0.086 | 0.098 | 0.092 | 0.139 | 0.132 | 0.222 | 0.296 | 0.104 | 0.206 |
| 79 | random | cos | 0.017 | 0.018 | 0.019 | 0.009 | 0.010 | 0.046 | 0.078 | 0.016 | 0.105 |
