# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly>=6.9",
#   "pyarrow>=23.0.1",
# ]
# ///
"""Portable, prediction-only underwriter model review.

Copy this file anywhere and run it with ``uv run`` or import ``build_report``.
It contains the model-neutral report runtime and has no repository dependency.
"""

from __future__ import annotations

import argparse
import base64 as _base64
import sys as _sys
import tomllib
import types as _types
import zlib as _zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

SOURCE_SHA256 = "dde01304924f85e16e62ffc1b1a3267de5b3937396392ed05607620c69cedbca"
_RUNTIME_PREFIX = "_portable_underwriter_dde01304924f"
# fmt: off
_EMBEDDED_SOURCES = {
    '_portable_underwriter_dde01304924f.reporting._underwriter_styles': (
        'c-'
        'pmF{ch{F75_g^!6hgVca|s1A2)H(VO!B*16FM4?fw{wqM#+pW+O`iCC5&SzQ!JB53?uPIsA}FigJ?MtQXCVk16te'
        '{>}$=udlDaVRa?is5+G_BVT^}NNT|kyyZ`<A$cW8yJw_nX_?WYDj7MJ851YLTPDa~zWw@dv+L{YtE+Fm`p4hD`Sy'
        'oD{qf^hKj0o;5P|fKsH&Fyd__oF6_tpZbk7cKLkhmzw+}#$qU~-'
        '&Zt@TLT}~CDrX?$oCjV2kmYrK5&GTfD+$qxG*s=_0>(x5@u-1E30+<707H1zaW)q{P6;+96-'
        'KF=dd%f>YaM(law)A$sjHhJ2yoVnv_z};NWTH1Hj==Iwe80ZWXitioDo3WWEu*<UOGuWod9qCO36-'
        'Gb4j<p<YZ@nt_{@r;I-'
        '!iqY_VcC(UReg&SSkxUZuyz<DuEp3^$S7FXmISSWO8E5zppp&3bbbIZfgCCn|2WYTd8k?`>NkgZa9*fuI>bHc@lf'
        'kY#;#Xa|sRZD>u6Tg<Ihijr)|ol9A4$UJdLOCWVS@v^B|+JcXqc-HPWWU-F%yI-'
        '!ZKA)1$H=8X3XJqnMnxpgmOt#fIYWPna&8n>oqz3}e?At>j{qDd^vt#_AKkpg3ADkY?pCN5a)5o2tz_X2_w!4Wud'
        'Rt}Ba+{1dbwQsuBrn+c;pKg^r#(_BIC>dDtQ^34;DM}7CH==^)AIZ|N~^MEB|MVCPbTC>RP1=EcdcobN%4&CswXC'
        'L={xk;Ru`Dp5oc%^K$4nQc-'
        '(wZpMN)>ElEV{x&SA5Zh*%r`BTBm$FFJnQGWh2teukUk8D>l^3T6sPf0^d@Cqh)uGDjE+o}XIE5LSB!pr&y;jRzi'
        '<sMeH(jM+gsZyh@MXW^YWQ{?AiezG8UY|)*6+AN%O8TT<qMg%7+qv%~tRdI;N3{if8-'
        '8tj&sWIa__h<2SP8t@qruQst;IKt<w?G_R?!sUcCc*CvV37J11F^00#N#Ak{(5a{FYT4QN#_FJ&+2*inrjT@ocrS'
        ')V`7M5k2vS!{bveluwn$^NHWJbjtzp&tF`}*WbI}N-'
        'I{!7JLTqBk^n`mrB*@B~g{<4Qu5}soU@NJj+=5J&4nVSBWX+H~fGpiI&QDX5gcMb^wp^#JXxAJAqV0u!5rE9_#=`'
        'NJoM;7TE0;;~cR)2D?o{!E0PGrA2zHqL)M@Sz60qT4JJB$E9I=OU0DTWSE~@{Pa3Bad4x@h8?XtUsE_cagLIkqXH'
        'r-'
        '7~K#sUK4~H_4JmtC&o%EjJ6f{%i%)6nZs~r5Db4s=dH>nEZA!WnR|N&vD<)Obl!MshDTgXNo)X#ba1zji{Ur{Yz$'
        '<+yvD!_m|OlID~2vWfDb9MTyce*nXn%)I)-vCt1q1BKz?-'
        '6npww^b15by)Pt570*U?Wb}qdcS@b>D4uPdcXS8v25@<^N>0r2PThLd?Yt13)p4Te0&Et@s$cInO0r&U!`q4K7zi'
        'lNB>_rMdUEk)VgfkCljLX3qIAGxMeRfm3w^dbuaw9xNZf}x{7Q3q@^yi!xg@Q-'
        '?gvIP5fG(r~^wcj=hMpcFe<|8U3h#aKV}(HFe`AiH-D%tGDKrHdbwiljF^ol&;~z1iw!HxWg<wydogztu{f-'
        'S*<SGp*p$K^B_K~+y;45hjQy+$LMl++V=+%-'
        'w<S!EKz%J9qqcbj8s)PmKI37y35PP4}<d98X0uV=P;FD%{qLhOB^(83xL18P_)o6uYZL@}FOf~=-U|o2X8-'
        'h6SM?&imY^53eO+vPi4CrW3d9lkYcMdGSD%&_Gx+t-'
        'N=U_+Eju?}~fg}3r<TAhyVBK3j#L@we0ay>=wrUJt<purr_;b9182GYf&kNoyk2oH{RL@Z^=z+<2l|upfQ6A`oC8'
        'D-$+d?s`z8q{iNF%CkqUVlaz-`F6?Te~tbg0jLH>7<dYO8wV`h_C(t_J;e#@I5%X-'
        'j2wE17dpPFf#2PC(u{?&KD#kk1Z~-'
        '(ZE>%??+YEL3FNsnE0zg=f4_GPy<8auD4$=I2NmtU(a28EtQqDJm4Wyv5N)E+&&M{8qZgb`A*^RZcXm$H1-'
        'A*01ZhCIO}(y1WFtEvoc!G&d8+3>uDQM(%RvY`zK|Bqu;4M`@rkPU85Ns&Q!fe1e9Sy-'
        ';XrzLFqciN_bTZy0UnBN)1U?#E@9C7=5y#?7SHnpPxMzUJ|;kgcJAraNj-Is9sNV6EV3gTpgjqV)raJ`lS{@7^yr'
        '_NU82Ynb}{>N7lRQPJQ?BbS83&2IAJ<*-'
        'OtTa~XusBW1XsiHao6A30KEH*lL3gl~Bf7n{g&DhJZoAD>Wf8i8~L)MmSJK?IWl3^L_pq{0Vs8qOblQLGc^iAG4^'
        '!M;VV9kLap-'
        '<ZDo@jAYNkb_O(o1gX?BxhtX<G%lpOL84Nl&KQaR+hoS124pu7D2Ufpu`c_g<l+MZv0(iC47donBua708I1hwHoE'
        'd0htfi3dyz1Sa2F=mPF@bzh43==Kow$^iyjd+IX8Ps(2F;!Rk!ntF9uMYexHqbjSByuS`F-'
        'Y;CKYylRgRM5}p!o&gsGr{i02L-JgCTlzSo9y-RsR-DqjvzqJ&nydOtHD9-'
        'zE{7dWNrn%Ru%Zgn%*iMWVbIB3N?(m0s?)vJf(&C0#b007o2*ITHFqBOUdThP0mFl2Mt9wl*#@UPO-TxSgwbVhbm'
        ')WwA}&Rpa~8HOkjmD;XNkVLoTf8gz@0yZoC;18t`7tL;T7q${1&xf?9fYxlncrJ>@h|7bk4;0!UwjL8r@FK$b|BF'
        'T*A)+_~ZT&{pyE5jev-'
        '!VX^BuE9yB8`#cX1>Is23FRq&?Z|w^XOH}zbd80PeR%ohRWbLmJ5Dl}Ef>o;jOs);*5v**iAmqCPTrwE7)$Nu#-'
        'pydnW?>4Z?6pzys&AX@tV^#nKH<GEwG~Cb;ISf{Imxvt5GGcSF_D=Yiu^A4bE6gc>!6N3T0O3T4QK;YYTm)>0EEn'
        'Ehrj5PO@9oU3skoNE@chOxOSY_kXV6(lz-aeOJ|Z8{@lKU@#4p*&zO6`3K#@Su_YTi1gzHL4S(mbrXhYC%3tC`Hc'
        'LQm<tyKW(`p0rUy!^1ZUGznHDFX2F35{$*z86H;{A+QeT8rbn!-'
        '|Z|~iReal?!O`twt%VeEC=g8vC8Q$N8FtMt&T+Y}hk={6*=D2N9JOo6&-^=r`M2E_y%cu{YMpr%b4UlJqk>F)-'
        '9}Mz+k;#}=y1R3$6dYNp1}1?HY7EnYh=%ICkpd~ZMg@E7p(zxzcA&`_x(PHjXzq8)INg;M^bu(jnxZ+Q7T5OHta)'
        '@tuNbz|-AYdO{fAFflwi}QQ?7mV*jj(8;sBsvP(|LPd6FdRw%=TL6l}-J%o>-'
        'BXX%R`yI8*LvGb)jraBMCpwg|qD~oqRhYM-aQ3dcUo-dZmgEwh<e@WbdOWa+rU)Idp>iqo%$F<Xmw&}k2Y`o5Y=w'
        'MA^Sbwtm@)vK5(Q(F`gVEfP^+UkUib>$oDWlDvecG+lNP%GYJw~GzUwC8Lnv2>lcX#b%m$>g@$GuI#)q{s=rafch'
        'n;MH)!iq`wPbU(|DsD3I=ZydNcZB-FbL_S~klmNV>`qMHhyY)q$uSxUXjYXT{cUMSSAg#gtyP;FT4<x398qWyZWz'
        'T^kAg!IOf^po+ptKcu~(aW9>*8V+sAfdS`Xjk%U7BF{-'
        'Vim*30>Pm4AY<@io&A*?PS1`1Mu}EB2LdKMkJdZPCxcIxt`R^R%QRbUp7X4sEEW>u<Pxa;KObAgk{qCp8JkN|LY$'
        'q_Zt;7$$<zdkv3K(^0$E)?=46;aSm8d_87X9Kd1@dXLKEfeD_<7pRT}6-'
        'a0v0^_T3Fs}u;LsZipB)TKi6zx&4P)B$*qi2q<QbBF=2n)=c9p*J<|NRUpgT4_lYn6;hoP^i^#6&e%Zvb-mgeG8y'
        'p+X?HOR|(385^j#QJkShU~Va8Qg#CYRl@Wh4vlT+8j=2RU{LpwTf2Zv@ZHOa^jGhC$aw*GrBSWlMcP9>=Yr-'
        '`ZWRpCskPc%lU<5cSpAaOv7GRpvLvkq%b94fzC5NZiw>0;3gKrZ_SoCd@D(9VDypK0wro$IIJ&>Qcd*Wrt7yp&RM'
        'pi-LDdoQERN@g=E7Qh<6r%j0(7F52aDy)68u#=qulr}1pWtx@WPq'
    ),
    '_portable_underwriter_dde01304924f.reporting.evidence': (
        'c-qxHYmehdZr}S?X#F9O5|1;pKn{VkUSxK*_tLm|WqS55NW&1+w%o0(E%_vQX8Jn)-'
        '$T}utcRrTo!vtYhn=>nip65FNEVC5n$2dftLwIC@1E7=sc(yYQhutoWxXkr?sPovt8$z4P10?ew%k737JZSl<*{k'
        '|s=isAoxR)^$G&X41e(^}zUa$jSM@zqJv7^LpA=AWS8Vz&scHbz-'
        '<DmORENW<FRu4x@~POL%A~pepAxG4+Zhxq=n3|hH1+-'
        't{;Ipvq3n{nX%EG|`dhi3!^b0Hrg2@I&1SQ+vt8R9l04s?`cqrxIdmjo6?NV8MPC7+vorPWQ1rL5{-'
        ')XQp?fS{6xSQw=uL4v!WietW%<_=_91H{1vdMl>i{8D(|nr4*vma&2V>Y5x<!3DTqA{4?TflEZ-CD8<U?8PWtIN!'
        'Xu+wD-?z<|yN^&{s=!BG=XC`{M;K76Z(sywT+rL`E-'
        '8RoNB!wo)LZxf|37Ze&dwFdF3Uqv_thp@X=qnx3H)zX1EoHd`KE!^by@ebx%_^_xh(eiwmg=#0HI5Y&E|AC;pA`g'
        'eY1f&<!%Qg)#cr7am-=<p%X!80}-'
        '&>HT$Z`i<_Iaya51Ev_8AARp0NM{!dlC^|0(tE%22=VFPV%nzq8uZ_A=T6h~e3378v52Ll4Es!c_P1jH(#O-'
        '@?7Me1Mu8F{&B^X||z{jEUu{0n?~-'
        'E`e0aOMw1S9Ju}S=)Suj!)QSGW%RsH@AHb+&o*KUBc>W%dUe#UIKHTSTjXyY*g&`yf&~+*VMH3p{lFH>5$`s1(Nd'
        'W@8Mq<vS5GOXme^|mYaiEt<dD`&CCDF-@Lqdo1g#m)A_59`5!-A{FuLgck%Y)CA9mO7vG8+mp^~_>E)~Q{N;aLT-'
        'wqXe|XMHU!A{xjo-ihm;4XUWf9ao|M2qF$BTDw^AGR-LQ2m*od4&~7az`lWOZJ>d;Rm9w^;3-&-'
        'gOApM62f+~I$O1erYyumRTJmpd^7Elz;nEW{$Fn|=AI-'
        '1}`pyd%3BtgiV9tMVG0wFbI>I13OefA`_X^A90vr5?4iL29KjYGs4eN`vTTgVajLs#|KMs>VEvkSp3hkn7dYAO3v'
        'Ki1w$8w?86pfo%O-'
        'S$Ach3aIyCX%UeW0V(0)^>CJ*<*(2GeE#~k=sw!lj~6fh`1al9$BS2F9mzg4h2#$@s8QZ;cSe?irmwE6ebwLTPq#'
        '$}0(^%Gm<sUjq>(hTN5cT6S{yr&EOWr>EWbQ|^YSgA6B@Mm?~v*lRzFiz(2yD$P(@fIs7mMV{p)uhqZ3GaV+v|SN'
        'sTC~5oI-?u!2g<4C~d)_b-'
        '3AczyBl;+zlWekN^)LXbI`9g3<}6z%Xgp`!R*q7nqccirp(;9g$*_j&%q%S#VxOdVpQ_aEN<@cR4>D3!mQe-Ltiw'
        'ri#BbqDMF#o{@%>414#WXxsq?c({v+1c5@8}lobJ*_^rr!qSepTNe#VZ3Sp-!J`AjC|J?hjIzF(BemE`4j#U-'
        '(du&V!s5#X#w_9+ZK1?^AT5r9O+WqW>;O`&Pm0!s42(e0ocJzgE<PV`_pk>%C?KeLN-EPFWL$;dVZ>_9;ma$styd'
        'N7F2*45G7joZ_52X@4y7HMaQ-Q8~c!>b-A>P-x8J56aKzFwKeqLp#g#Iw_sf(1LVi%bJ;GF-'
        'M%R#<Yu>n`J`AT|C_vpB@45L|BOzHHbQ>_qRUy?7uO{aABZD<3h6C=0%RBv{8w}j|6VuEUUfIx&#U5CNJ;i{U3@C'
        '4J?=VWhZ|EwX!B&QmlSJ-X(mT8hT2b(EiO!tnkEuGu|EoM>7ndzo2?~O*PPl-Iju`#6}~C^0_)1oI$?k=6VE1Pqw'
        ';J}y*`Y6$?8zUa}>2VK<aZrX-)>Uc*uCwwq=>P8yQh7XDdKAN+-'
        '$ObtmSY)^xbo0+b4JX?W1BCAPSFfm7y|D;<?@CS^p?)*wK9VPLCtmW0dDls$5(k<An*Z7sy0rU!@@m@JML{AWFd='
        'ttP3<-fo#ZJ7kNZjYOno3`5K#g}qG*rHCseRjGncg1Pn=Tg7jtx!&9kD6U&wT#x&Jqg=sX!1mbd|MSaH3-'
        'sb(~W}uTC6iFa`irybHN5wV_RLHl!9~g4l0!V8aK7e#NX(;s_CX}PDip-(T9^eg590(U2&-'
        'PclLMxxdeU7b&i(yVkWBYShjL;c!iT_IVy_Kam9(`B5l>gGA_phb4?YB!{5FDoWkc+BRK#A>wXGEtJ9V~%lyWY??'
        'iO8sufh=d^Q8Kro57=_Ksysy^6KBFE!S2SMGP#R{lr(r*L5AMoQWO%?;9RX}@ojmC21xS)L`&z7KJh(62?+l|a!R'
        'cnIt@5X0s6yltCyvKjROExJxwQV!nIwv_}~2FY~h4+?rF#L6W|?QBiY#zwoXS|RN@htsatb7fix#L@InUe+sTOi)'
        'qK4mmUgM~-<mo4t~_CPlp!GN}QYN<6Gb`*YRbYFa!4^LTsOP$x3_kP%uA)@KET$DODcRjU3~i*A?HO`lX<Rd-'
        '^=r?Q{99&hH(ExK{VjHWY%(&5zg$#t0&AQMCzu&L$D3BBB*&z%i)2COdiBgohiG_57bzUulbP8D1n_YZP`HZ5U^x'
        'g|zbEBY+J`aN_h!P*kQwoEAyMi0>fXA+>ftfK$0HPrwXuz^rgh(W$uXA7VRZ1L?971oM&{<}Sd#uh+Okq@b%m<|b'
        'G_|MeyCs);(yQ_`RD>%~{ww!B32t>RKL~eAtSvE|21#2-l2WDnhdRz;|>kCinaY3;Pz-LvZbTHNAPhLf#v)WKWAS'
        'wS<m?T7kM>9(Z$T-'
        'uqC207XICrRsjrJ9xHK$q*);*furdbXg3Uty?#qKH14`q8(hIw8xP?iN}BQy)HK#%}%HE60Ud7%76QWXpjGpH+d6'
        '-VpL2h~SRQG#s#vsxII(NWfKVOz8TslMOip82#dTJc$WCDhpDIUiI3%W<ldf1C`Q#@ce-!1~OOdo%?_uSWf-'
        'Rh7)-A4(=zTNB&aX|Ufd1p6)X>`mU9me}_6t201qY0jyoMm?%dC$uCDEFQb|7970l`OGpdh-GH$8fXKUq^?EFv9s'
        'ST0qi$t`1b4>fH3w<9lLPl3P(FBEBhxgGYQRC?%8ijOUV+aBlR26Qqsr{Zbimv<pNsrD?%Xg!O}+jB;d3`Pn+m(w'
        'w3-'
        'u8j>laZOLcboP1Oe+|{Yd#U24yGVDux^lqkh$}%d&)!DqAHWF44MA4R0Y?6P9vYy6M3eLFfmZDP0C}II~Q!XYDm7'
        '$_qI|k>U@vF2$cC)$(#B>-'
        'fklb60zG)wn)DcJ{uS@t>&)3F$s$o&&)QVJ9UM$y3rev&#6>FtWcdj~?7K}Wvy8PH2<$+Hsk)4wcGob@`ErvI%%)'
        'tk#=+zK|oB_Y>_INq~2D~9jjI@*>_x5ijx(D8>>kW)Vs=cV-'
        'A;L$JB50ryIizY4)#A?UA|$Fjai!oeHRC~IN`+%YrI^Nv%5WbmZ<z&zn}S5-'
        'z6U2H|EvrYt_Q_cO6<XMmtQ#?S0*yK$iNsm7=aX!ldd3Q@Tg#nfb?Z2=Pa|vJ(0-'
        'xZVVxhchP+GKH{Rl0u%)XG)2(|BLy5^#1<<7Z9!Hs1K<r<Nq;$JEt)48-'
        'w`u5DwnwtVmA#^<ZD`!YIE&|UZGcQ?l_xQX~u1LX*rShNYda6aSZu@613uwtwm-'
        ')XkxK>h}rh+SPj&hpc0<gE$y)$P~`@U_P%e^pg5+8V$)Q#%7hJQx8>rab-'
        '5%!DjiQ)b%Jx$@rXb^Xg5FFm#elXW7Uj;CIS|4vPEQo#*ykf9*78zcRA#1BHe3$QxuNU`!XyO>}2#i=HjTF;IZhX'
        '`*r4inq|hF2b-X+JiM1(NnL+aSN&;Q&K}gEb=RTFT-'
        'g@S5{wWKz|dQ9Iw0pMx~S>{MvjLN(p&eUz%7#*2ueg}@jp#fryVpv<3ZNS)G9lctqh3+<gUVFm^)!JYeqRagnhHQ'
        'TF#Rz*|#)@<e#uuZ6DQ+d}f1Rui>xbot3WJvOEgsFR_9}3yURs{luO!mq1dtJTP+$U8r}AYLs%bq(e>^I7^8jKwE'
        '4kFePBWOP(*DXN#h~OS8w!sD*2oxo+xbHOA_|M9dgP2#PQc+q*`HPoN>b#Ye&6^Bf=vQ#!#$Ari3~j-'
        'u+DzS!eI4IXY0!Y#VfL8}7QQFmLk?g*2ARuEv6@(&>x>*L~YW!rS=zEty=fuh04kd^3&Q%}Sg$ng?K1sR5{!H1?='
        'aSWv=V*rwIdU)SvOc6u{SIXyXB}Ei5auBPPGqOch^m<Q2gs+;N!?C}!yJjTc6UgA67p)wUd?gfy=;v%iBi^4UoJ`'
        'Im?mO?O#&5ceC_C<YnsAW9HXYt^%FK1!EQitJ=Oxm_B0ZplpdI>0Ccl(-'
        '4wFkD98ai4T>XPo2%$N4<iPq;zE4dW9~!|#DFJ5$1e{hAd_a!!P-D`-#Mo-'
        'hDhm?_cQ&}w-iZ#LDY|P|_v#PaMl18KsL|=$d`4$R2o0Q(Z&oIPXt-BzniSy<MZdWX&@M3Y&R(d<_1}zy`lCZb{h'
        '5GLgXd-#ZKGLa=6~cwP+`z|E>sxKbXq8&k{DIJ*`xkdq16ToMhAZZY-'
        'e9)wi8w=f<Ul__#;#ca8+Pdw#5?=RR(1z9RmBi1mhK&UGfE;0J3+BxLP8<T9?4C6!q3YLN8o0zz{4CtOC)@!YP%i'
        'PQP0P4WaNrT~O}g-c_Lf<$1`l=_+Cn$+MQG-Bz{qA&E)S#1@)GO^T_`OkmsA#MEPyJx1r?AFy$SKh{3yDjGe-'
        '4wg-'
        '|0c`AEcX^oEF<weyE62mXJdWeEw~l2IiG~4C!e)|vW@Af^qW2|eF_O|STYYiBslo8W3k_5iGU1@UWz2q922qqV@9'
        'i6$;xL}>Vsdb$!S2exks2l9b>IgSn0dHvPW86a6W}m`w4t%T=7{bJ=uEDKTwc2iNKM%!I3{J06Pa$qFFE~50hjUi'
        'I~NQD5OGgPiGaT}9%%TXsM3@ql}Dw7?gSPT;*>D!2yc4f{mTTz2M9>LQsoh<=!=Zo+D%uW3z0v370CBKYHC+{in('
        ')IiN%#sfdOt@DWw*2Ysah-WBvLcXO1UT4RWRa4PhXsf5q4&;2!pl$e2Ym<Pi>?l+nf!Coo~275O{j;<R(;8~&7SR'
        'fxR|_!5SDJlNP+?Z(0OSZ9`U*9E$mXyntSW%OPEob@`0E*CPkT^m66?$yeNAs<AfdkwR~s0ksb+)ic4Jer^0<;<t'
        '_r@+Pgqe1whSAM(mDSa}KUe9X6MejbFx!P$+9}CV4d^apdB;hdA5#>Y7X1(6IjNl?qSZ=+pYW$B#Np%ZT*PqNT8W'
        '|S+rEStL0kLbb*Jx<liDMA4i(+<*HU<XAuYSc$7FryJV~m!lMQvc1aYP__Gh+<qiH^eophKQRA3<(cDsEGyn&$_s'
        '9Cf@eOdNR);1UQ@b}{FPw-A;NMO8E1=bhXmb=In4MwGZn2F>c@qUc04l#Xv&1n-'
        '<><PKY;oh%Ia4G%^43%0qE$S%sirUJKY(KorsNRVd2_LuuWSP2sKHNAv^?W?*^p<}@6uctB<cwHgzb+$l`mO$`>b'
        'jU!jYZc=I+aBq_xRQ=jH4xca`A%c4)znhjTjS=QsbXcZF4X|+_;5|i<;ZpS%)Pn?ZPkFCih2xCLH}uMJaAWDV6mI'
        'Ke{*}YN3awVrdTWaJ{h>{KNw1KkDwp8P%HZ6(7^Be0iX}CEd|st&ea_OGw->EaL^ZS0o-'
        '^BM}pXaI*tUaA$Fu#LvBr-8fw!g!F;o_Ev8BoWj0FyD8*%ef+84-'
        'TC_>u+>~Hww=)(lZn||9n7lWn#ueKw%HhU#Vxmypp-'
        '{t*0a6?GFd$i&v7=gNvu!z^NCA<L<!<#;0UJ##x8<0=mynd;#;uJM=ujZKd!$4kqW^VUY<?+w=AUv12m=y-'
        'u`)#yC=-8q5wGMp{Bd!wfjc-'
        '8qF<vo_Gk<&ZXRh2fHWvt&}YdJ{#$JBo^{2pw8?t3)cI}VUwwn$9_<IlGi39dI4Z5I1MTh7veC0ji;xZT;PIjbYM'
        'SZ}#m@Y?+%>|9O=3Z2V}o7r*p0c2%NabQCwTD2<WNIE35?mfmN;>^k#3~xAu?=pV>A}LtWG;<Kln)@t!|fMW4f5s'
        'XA4DraF{FTS+^5!-Rf*oxm(@bQI!KwKS<B|lgs@SU!lu;l^Y{9P3DQkAv}UGxT-'
        '#$dZE2LvRoWY>{&oWAc=Ak!mN%40@!8zhV4A;TBN|SZt7hySy&K$nde_|-'
        '8vRA63O7*o8>rp#Q~(eIs;ugcIV^o`NAqB9JPw%(s5}(x-'
        '_mBwuLwrRjX~G7%!E3);ZWWu>>>G{@oZeC;(ld1wsrtS)2wOXWWw2XivS60>kmAKPbjyoZ@{GduK1A(5rT(lHsfu'
        'D9ckq1n$;eFVGp%!_wH6U(#*c=$-dqV0p<@z80V>ehEksYvmDbz-MYEf8{XtfHO7VNcAtCF!||uQO&CNZycB+Qa$'
        'Q$)zOaLHpmHlba?bEI*qG=rT@F!YOpbq0p&*Xnb0f4gEpX(g<(1DRh3qTIN>a79WDvGn{TI@1X@1_y;c3^#w4-'
        'v^tp@yrfen49Bz^yXL1WJf0(NSnTflw1v(uC6%wlSsbK}wT2`R$<-'
        'atjKt>c;TT3Bw^?Z$DUjKTro=Dk*QT7_IBm;xi=FCvL9%L<Va~wB5CM4dNkL2!{4<`PYz@*Hxz||p7IVE-'
        '{#)|tUbT&RjPX>$!SgYgy#5*8&%s*FU-rM|Ke(J0T?>aEBMT!B?x~5_}!|Pf>hq*W#<;{D8+(R<NJsQKuS^{DMbT'
        'oZ<5Xj<oAj4ju2(~*gUV1%zA7vXy?qa|_rUUL#Cz2cmkdXuEF}hNa*8D)MN4HXOtQI2CDk%FL0p-'
        'Gjq1?;z#*kah9Awa%gR7L<^i#b7W6RiCY~hYY;GlE`dV2vTidoD8-%WZ)&Y_e5W1ZRq7ryCq-'
        '1?SpQ6A{kD6Z;6_SR(M5{M5ZzO=Hw>2Et4@dubVA%4Xdai(RnV`gm8i((x#<sq>2OxvnH*~pT<z2h~6*L5Y7vk&Y'
        'F`&yI0hA*4)*jpdLNN3!s8%WPOB7FRtI(jm!!8hRW4RolTy%^sk2+6OvWyFvZ6D0`%^uck^^Eg$Wp++$|UmwIm)l'
        'J2T)j!&wcx=NdAyUx}iZeRg2_uyN=MM>{+zWZMBMXF^fJ09Du#v|+<8hA?chU>@kxBq<TH8fP%tc8xH%srYI}Pxl'
        '0oB^(Ga=fbPa|eI|5~#ra?%!zx>UrEOHgv>WhjsFl|g##T!BGO*FZ9L4@Nn=pK=k#!}F_Lg~1#%X*BIH8C=-'
        'hie=zk`ay>aQ0AD>BXxreGo2*rm%8~}E5bR1Qsgth3^uCOBPn+at48-XOgdFdlkPVSgETtK)(zxrmYxl+)m-'
        '4gpg>tE789sS=gBTWu*JPB*0;!AB)vq`zY4qX3D;x<2W+W8$YHo5>^ot<(Urz!$6c+ljhqgtC9Xd1^i`ItI=CF(6'
        '<T1?Op8x4Gw}$|cR9(;QzT8bn~z$zMvbi_GGLckZNweYE1I!;t<C|ZKjkUg4;HF^TYM@L8Ad?DpSxMadmGBFOt7|'
        'oLW2H;gUbNeuZ^+ZUTLdHGc8)xFQ(w8{*B`0dV}z640QO2-Shw8R6j7_zu&2TKvX^LR6hX7JJs*L-'
        'm!k?o$GgxIM|;$1z-1Ie?*jzNYPQl(z3QP8RRvJfZ|W17b>!#DG-'
        '`LZCdDS!_z17*WXo{a8HH0hct!V%PY=}6^a^-'
        '*kzgj!x{S@IM&mU28E*gdzgf;6FfM$ld%V=B(X8w%M69a&cB92mu_9_3L%M<=b=ucXAQ&dm!Y@@$Bz)Wry2DJF#|'
        'E;e;Jl5wsY7Ai3g>`@)t`{_rlI12i;&(qAN`E(cW}o!G}2uoyRSVJy9c3$RqK`BYqnmeSD`)V9Ap~2AS?MYm3iN*'
        '~GY}xr`ijoLsI^Zi*N8SpG1}LRU}%dJnoAhs6(=Ynl-3|D(4oFba^|&&MwX1<^y+)<Hi3QkfWB(w*Q$2Zbm0u89-'
        '1cQs=g-ypr!B6KS{-'
        ';_^PR76i;CS8H*!`;k!BV_g8+cT&=?6IxSo!RzGS=JE;g#jzk4DKCL>T{)A1c9k3L?LCIDvF}tNJopMajf>8SKCZ'
        'L_Bg4vt-1nKy2{ra^t{nsveZfJmXm<!UO&;-'
        '4ztj`DBj{%spjABMoavFg?J^eh}%V1f<JG!ZGI@s(<K8}V3>n)dXVAJ{g-'
        ')iXQI#MLLC`&MsCAsM8daVxP7hbF5Q@v0WXqM!__hx8^_UwODu4KWOA;XKo1kHtRT($W)DqecG!PiJO=>?vU`Vbb'
        '*1ux#-co$v^S|m(oR2M`HVrzfM0_ZzsE<Il7Hg;W=|ZShHD0RUE<phb@E*B!wdIX4#8s;`nuTct7G~cUx@-'
        'NGiR+03SX3D#F3_=@bt|oTmpWn6K^#x=0c0Z>J^!1A|T<h3Xc_WZy>pt1>Uz35($>^mD~)dJ)!t#t?YdT9M}5afR'
        '6Gs9pyc2D24Msn<why9s3A2BbbhI4F)7@Byv-o>O>{$q^-'
        '@6jGcZ_m2W(pX9XutXfS@!G5<Ma#oavMb>!U1>%~eR<)G9@*7O0IUo!;n+(?}&1WsaT^rdqg6MdBM5yvz9uGBE%P'
        'dK)5!cxaPH}Pi{ncs#WwGPa;8?Z{<I3ev8id=P%X+m|z*l0m#6=h~9LiTZ<@8Jmw?Ez$zXXN0BKU2w{sEi#V1=oN'
        '}#<(Hi?eqf8T<gH6tg#aUo4Q~gI!Y$t%wm(_<fD6vec|KCGHDE$8*CrJrocg*o%^JG1hknRfUq~>m4w-'
        'mO+6nP87u)kVtKec=)Hj&e*cudefntDR9?4%Xte+rM)OzGppa>7YDoe;mPPHRcL}bXz*pA`jo_3M+KMx6&k)8@=E'
        'K0<h%uMw)1z5s3kH&PU>&2O{X9Q^Uq9{egKy{QsUh_!U|0G0weV0_r=+`hc6B;{7cKyE-P?>VB{+%O-'
        '|#a_{(*Ic-^K5YUJGZbjj`EWHnPVaHV=YB<27dCYsIv6skoXUPdH?3l}M&6!MoSYnTKBA+8^s|E{E-2O~X5uD-HX'
        '~zwHg;%f_gBkG0Zu;BF}dDhQuYD|QjZyIf%!54Xq$R~)_Y^LJTtbN2$)T<l^Hx>}9q%93?GAGg|f=^WmhjHfes0K'
        '@Jwe3kXUK*@FRDU<~Pzltmf0(<=su<_f;;n-XR`cZjheh0G-JY-XtoOBZ5XVacB(i7R6GoSUsXf7ld0Qm(p+%=d^'
        '_4Gv}tK1+B?=n~P{+5)?^)JFvrbmo>12u=KdZZczLD6l>8fbT;g^aj>H4_Qtv4!zhu*<djf{g6MqIg>t+w>2a*+&'
        'P$ex}{lvu=RF<RIgr^=1K)_n2|RhmXtQf<yjRuEH&*5bG;`Y$R8zj&8-'
        'p>Kq@n#phh@%Iiqfz9EOzn>uqOy<r6Xr3M=MC1wg-FkPso2tDPHBM^>N1Z0S&3kY}{0@YTqeJ4#V9Tqnjtt>L0un'
        'SAPg`4-yDgMSlynFm1)BpjN=FX9^D9TQT8(|97!*z9YYEE6^KSDASAls6rf{LC_8Dd-sLaZ%;?)YSO`pGszFq?rz'
        '32|8y1NTv_Gj3UkZ@w&73n4t8ezG&4U)96gG<e@e=>lq1R~IQ^3KgiI8Z&!NCMgR65t73^4nWfW${O<|m4<SCTnL'
        'jtlebd*uwP{{%v8y2o&}&P7ZO6s0&Vg8Y)-'
        'G>1)HP$?WQe)AW+5Jh`1F@+#?|kfcj8|!L+5w$0if^j6%VX`2Y}P5I!7TWbf`|i#-49so0}^nCElB+NQ(66;7=-'
        'T?HL7gM>C!@{-0_j9Wo|B>L}}yGbO8&t<VmJFh^|ob%Q)AcP3Isf^+lq5f-~*9i$Ul89-@&iEDCp$u51%-'
        '2s#6Xk%(ZTfw`gMVgpWgd1}cQ9<I8^|v6$S0pCTj5rd2f>WL#~ckD;&1ysE_Ud#sof*5UMn+%Juod+YH&Qn)<IVI'
        'rM&ar+X+3hGn5Qoz~BQL%LUI~2pZ0zGfqV(ge{&@Cz9vI&Fd0R^`ZxZFu<gNQ9dGi%1^x>z|`%{aspjLG-'
        'f26!HFg_V#M~65i`$cyvxP3ySV?G+y~QzJg_e0jon~3G*_@axLC&C=?XrPi~kEUfB308`#jUid8sd0T_!AD=__LR'
        'O0EjQIg}+#t=uRk#%Vh6{+<8Jy8`KFD-IJboF!L@wv;}X=1XVtTB!4?=rTFSzhA?j?sHLt&t$|}SvRA7!0rhNcj;'
        '56URQ(pw`{jsTzKtgnNcfkb$zmLzp68oPbHpp5*!w+Dtq3M7xBA^tjNTlX`2QhoVZ2l$(C3*RK##_>MkjX-'
        'm)Oj%E+&a&fX2qFE9T4JpbY4<@x1AGzbh$Ua*tEjyu&o1|n$V;&jmdDSe3`lHr`d#$$|aRpz1zCo(HCr_bH7Y;|('
        ';aqzZ2hi$*pGeiU1-S?6r8CX7%a-'
        '>}(bbMDFs=cdv2?SZrk(b|Q?u9KZb5d^T){45^r!Lo{w@}hEDzyyzt@v^{qZh>`bF6|aGbiATg)E-'
        '!UPXrN)^nF0FJ;dbZJM@}ZloA+Qia1M1%a24V$gINBJUf8A!lKwCd1yjr~JK=Q)C^EhI3YghevY+Q6{ljg8Yk5QC'
        '?AAH-XJkZmmXCgL8wV{Bi_I3R=u?woOMn+qLq=;JeupuR~N_ho_RZyemI}Fz!Y1O>sCBwD20(6>#o)ELFr)<qua{'
        'A|BfYwBaGgJ);Q|l&CB=MOW^c{WcY1*+Fz_Aji&MiOIeES0&}CYE9U^iU++9VaFcPwNbx;n&P?(89j5@2~n$*txi'
        'ROF>l|gqVJM#7o!o^26z&azuw973n3W4^{xhRCNP@d{fBoyygq-Ezkm0a^ADH9(`ClQWH9~!?RUGVx((o0rb?*sz'
        'Ow1_7@U=d5ek&(j7jc!r^_2Qm7XOg7pM3yc{E!K+@Ik4nrv(7ck1S&+efhzWB~XE5c@)9FZ&$Wvh74C6w=rxO@QR'
        '_FtoYax4BOip`WJEFA!0D#=K<|0+qr#+RQ?wuTkx3Je|izexJ&VsXd3v@i&DOvTw5I?_h&ev_gRIi@PL<8fV3b+c'
        'B`W>SNB*U|qFZq;Zt`zRJKJES$;88KVRk+|KHa8|4U=YE6LC>ev1Nb(v7qlyRcoh_;tE{Iz8=-'
        'RVEb3d2JnH#ePKPyS-'
        'iASNWi)m7b;xA<_E7;Yj?BG_VJgQ7%Hz<?qetV%nle&R|t`TKkloZPK2a~c_HgdIoBqc-3xi^!lU-YETuHbra5iv'
        '-o@;hp~2f+Rg~2^ECqiz<{y{Ecq2YkKwa{mUONUSE8?I3K%l#n2$2yeCP*Yv=3<4iDn&a0cKzDwL;#`O2`3{EioT'
        'i6cgSbYCXQ!teM*m#eWm6<~oG9st7zp+|dg@X2y(TvBcdJsQv0D*XJn{|ftfsynB}gEOh76P40@1M(k!5WE7Z2b>'
        'J%!$!5gNOW>qr`dqS(&w#OV6leLG~w(Vb8<$f8EZD4*9nhZ67DGxEe%$(0=qltFS|gefjfiBXX3klxKl+$MbZy5d'
        'So@wTpwG__y^Uz+3mVgCvc_)zbep{ynH}6(uJ%`x{x1czvdYg<-'
        'Sr5D%>CmnyUn@at!QqYB6R87FL%<rg}DE))G0!pqoohsO#lYNKL=$&nINyGQS|pqTW^qk&{FdK~js-'
        '00$0ahJEbmS_Zqyrs8jLD6Y<%C*c*G9fKy&_ZeV>S834al_<y(+0pfT#2{k#KuZrHP-zOjq&ivR7=|qq^Yha$aYR'
        'YZFKlZXT|;2{hg9rMJp#(`5P|8R@8A@=E6O$?tI2+xWoEj^biR+mBUm{mbXB5Ynl72nm!C*@Kq#uZ!I`43HHht_i'
        'pd#FS{#0X*}l5$1o%vQaan}R=o>I&ig_r8id+TSwatO|z+Ltjy5|lv^IZ3)<nh$jbB30v36>sTvu;R%IP()sFsZT'
        '|MHAD}ZcED{wH7in&p+ig`fj3fj`rq@;Rsp0n9U$v&YU#<p=L;-'
        '@a>EMnY{Ls=;XJWN~m~$A%zw28k+Ybt=Dx<T&#2IQ+${v?+J8oGh<wR{4hp0-'
        'hu&+6Jt(#!UZE^!krj*C8`&sRT`~He#R+d<yeK?x)kW_F&<pwp&4-04moKfNXA{X?&c%-'
        'KGJx=t=*82)9OIz1=G#;hozDIhLh$3V}^pOAqH%eKb3b{Gw8dN0**>tdvAfF@7;!9$~*UpC??jz?np@p=FpbR?lA'
        'cl_Aw7fv6dD^^Mue~MqW3&mvgh&od2G`dHMF^#j8n~awk2bY9A-^gH{7~p(A+Fu#@+%-'
        '+grRt&Y40VFOvFq0BwWF`B}^BT0^(;Lfq!#$>mx-P-!q-'
        '?loYO!>(QupDOJ&CJ^q;`jIV_vC!Xs5yjZlGuG{J)taXPJQwpAK$!A`r@V=Ae(#LJ4QA?K=k)Z5>4^;GttU%$Zhf'
        '95Nt%OPB+k<_C4QM^GIY_!|od&Gc(Tajt@lv7GtsgU%r!h)L2-'
        '=cywTa(A(hv@&sn1<O$$Mr_nsNAkB2q{I?^h;zF@!SLY(KlWc(xW@L{}9xgH_NJv3v|KQZAd)qd%gnn*!c=qs!bm'
        'VaevsIk%JVf@C+2Tt=uC;+YjyBZ#?=%CN^fd9n7~P%wMJ&DBmFqs2jWgz=)?|@4gx2AEBQ&f}-'
        'W`$6hVP7UBYSZ1i+Na%+_nh3ViCCl@i-'
        'PShri_Xk@5jQqYU%o&@yLdks*y0i4hy87>jKr!HZt17UioAz*jtR_J6O79Lo'
    ),
    '_portable_underwriter_dde01304924f.reporting._underwriter_movement': (
        'c-qYx+iv5y_1#~=HBdmxDxM_3JghNZbb$7;MNt%eF${sRX-'
        '61|(o#~!+G+lM&*7aE?Qyy&w!vU5ljnZpIh2>n<v&FAO83)!Pmhey(@C%s9T?fSO-'
        'Du5x4b9qku>dr)ua>bQ0<4R<*@L|8pem!a=BbAj-qWySsuq>6s#;s)pV^G2<5yTD8l!Pg<fmu@S@=NZCx`-'
        'tzXgSy+QgL?YfGe6ihWRF|=*no869yW;8S<$S@(e+5gNZ%ZiUpHxb$s-'
        'Ypi313Qv(s91@bmS0(QdKuWEe6Dz(EeQO*(t2e58lYGFK!u=FA=ee?eEm*+a7kiKDn1m8oZSEH0j?zknCNrgS0_$'
        'DHPN1{1}$tPAis>14^_<w(ji}~;iVl1(sy+=45%3cfLco@&5bliUi~t%0=T|1(KE6^u4*w;pal^O+`&r?viX-'
        '7S?(cg8z@xZy>CZ8WE#5A^7R|_S&*jUHRGq@Wdk3XB9SY4O?Bv6&|i_-k4--|S-'
        '=GBLLOEh$z8}>VpC}wFoeWD)@>^?WmEvSq(6k1D`b)9E82tq8Av_k31a)Y>N2xbkcUEh3#55e@MRI9Xd61Di&j({'
        '7JaKBb|%*9_EgsFmDMW765$0Q<cE(8o?+~2w)KJB^G?MR5Iz}k;aOA8_1;$|U<=GhhfG1Pz;&5+)!223eR?+g)lp'
        '{Xs_OT3+p|p4D-;@->bWn6w%$Cl`^TqTN#nTQZ_vkfQL*~bEwaOj^;&Guxug4-'
        'zV^<K#yj$xMCch6`<EWlnjJDpX#2Rnf7}(?8U^XA1KTVG2FFri@1Z^k!;zi4Od}#5@Gw+%IRw|!>LKtda`uC(Fw~'
        '59Z2>Xq``;B2s0YvVxq@7u*2*cmh{$JU;Z_`^&j6Q-'
        '6WYM0_?L<wHp_mX&o$FpOK6ZwfdHC%*vj=?0AdS(dhkKIwy$K7h26|0b%^P*I$r+}t*CVM4XgqXD1&vG=rYd`h2v'
        'gC>Ohpn_TRTcuv(r7$^|t$CJZgo_D}i6*`NG}{JrH&xnY2_thT8Wlp)KvY#8Mc4EeL%0-'
        'Hz*xfACCwA>59C2cr@2MXYl+P&lF5i}r*#J$JHk$Hu4o$@`KGsZlFa2Al3mS3Kd`xX+OKoN}t<1LgVI<y$ovfT8)'
        'Kv0L$-'
        '9^08^~s|J^?mga)~JtOjnZ+4{+!35mEY6cN)YM*pB_^ikG6Fx_pGjUOL5l4J&wBR!n$Bb2zF3*zU3OS+l!0CDdAW'
        'fAXN^j+N2wqb16Fj-N`61;}y`lYQ{$DSoYPg%<p^D(N$Odu*EiU+YbU86wD%N<sDRO^|03U8Hh-kAPmmt#MexrKr'
        '1Wxprh<Dw4e@F*{=QCN4<<=;I}W1R_<sgdm$hHayt~$vra5c#qep`c=U+dwU7x0prPepQjFDZ9y|T_^LNDFApovY'
        'evmO2G*y-a9w==+HoSzRYyTw+ES-$nJa+=nbJcGie3lw+<Vg=2S+;8^5xZ-ylLm9KaH-Pc1hFR90v(XI>LBf*wSl'
        '&w_m5ksQ*!SJ1<+~^1XTY^2j22unP@W%>7au3ZT;#KQUZDbytU+Z`5X_8fH?x9X3k_u9H_Gq+NVVFTky1|(ECL)2'
        'MGHX8Ye!oxu7=et&UMr`7%R6zQo0@yO?rTi}K_~lIZ&t%yEn#vOt?}!8bYgotLsay67e!KA4)7W0zaZai$s0pT38'
        'JVWP3{(N`n^9i<myhMrE~C>!{Yx(r?dE&CVf0p<wi5WS4=9PMr!d^(AWm{Ki`oGND)+o6;K+sQPd9~u{Ad9&Ywju'
        '#`RNMuQ964Osej{8ARko3%SOCh_9>&53C$Fba1--0hYdvoy@CaPCPq!{;DdA-q~<@H45B}{kW772y5cNU-'
        'ykSfoRf_jkqW+HtHxdLft3L@4*iTjrOm{SsSx`uf!8`=TdH#t9Gk{s9>^GIUIZ-ttW5R>t-'
        'Vc>k2|Fx|7o9={u7nupj=+^h>%u}K{GyHE&q;p&Z>IE}RO3#4CJLy=@PqX%R8<pfHn+3W%G`1$Of$p8Z?ga6U1dA'
        'V=dIkZ9hjkCC2jn+;4h>*K+8?C2Y3XJ>uki@UBC7R_GN|3(EnLgA#{+fy?d}dq7OHs5=-'
        'Gxr3W*|%^c3x381CNYlJ5K(;I&_hr`OelVFA2~3)Qf&nUr5fpAbD=5H0?cn<h#ZosL!l{+qxD`gHq4Cz9Z5^}FMT'
        'D{`N?xhiw^6O?}e2Vc4YcBPz{4qtKKsE=81{T+JVNpp=LQYp{2b(KeWAhN!j!&2iArT`%lDv2Q@u~5(M1VZ_fbL-'
        'r6MHsk_T`@;U!NxO2lAa3*x2=t{@h#<cCx3T}cW2?UnXUL!gwt4_b_H<<eCh5I<9l5%a;`^=A?ht%rs~H}f~<nnU'
        'R7V-'
        '?N%jE$`8@b|JJcet+`9%^>;cmu#im9nKSiz+8>Gj%Ok1Vl4ocHN3`(^<tP0T1{SlZT{OeYJ$W?O%r{P&`q*%@#S$'
        'YOPM>hvSf0z8KC>E|c`SjM>_~FgoZx9BP{!=B2#XHhweQ0IbveD8+H{4QG#ED1t`LO)vzFLR4`ID>cyI1;DczDXi'
        '8}0I=YsoVr9=36SX|J{kF;AAi&HkO83aN@#WdH=o;6rCyt&1@3kRetcjW$`ev4%7lNcH`h%-'
        'q7CN<GTV9N$1rM$Fx(=mazr6-'
        'd@fz;uf75^_vyxWAqzlYVw>k@W9J^Z2K{d8SCKa4c>68=3tbbB1n{W7wRtJBYT@$|U!%ZqEhFHhwmPkirxK$cIkJ'
        'X@X@RNT%AqH3J(qxK?4>7I7pYobdtLRwIrHE+6+^-'
        '{w8P$6t<9RReK`WFYPcl6%2{y(&`sV8i2y$SuF>WqMjA|`+mY>&Sx(7)J#h8}+b;YM;}4}h-'
        '@Xn#AN!$if~fmA&RI^Y`@{T9AZ@20W%&$pi2ww8PP)$J$p<jthC>KXYrmes!s(TZ%Tp_;KD$TK69BW-qK0)J#e2d'
        'Ik_cAl2@Wsi4f9LhF7{E7V$fO^OI9cU@J4?cqdWmF5DUsGw0X0_8-'
        'C9lkAxiqJGe36T3(#xpH%(QU8ch4J{T@jXfW#S>F@5~qRO|EV*_@QdZ4;%7`YKLT3kmeo3%@|2;Tr-'
        '4LhVnx?H>_P6I_zKYmo2Xls;%gqx6#M=O1M9QQ3sYqGUO{-'
        '*O>}8(49{)0Qn>@*Yg|1l)~T*G=*kHv2*Uyoj_>!wU>fkmB3q}H+!AWX~WFD4Yo;Mp;-Zorq4X#yx&s=-'
        'ZXFbXCa`?<nV5GSJD<o@6MBaL(zW_Z^|;#N4oYc1Lm=h9ErJmJn}QoyjP-'
        '|eC7LVi|FVcN7DFntcUeYMHZclWS|3kv;LLsuV&&r?+!E-16~doB>%^2HgpT}6u#pD;kF^M-'
        '8LeIawRTnW)_IZ6ltiKcp5C(9E8Oq+zZCmaO>t-Zgz>T?6dd{j(_)V'
    ),
    '_portable_underwriter_dde01304924f.reporting._underwriter_html': (
        'c-rl~+jbjCk|6l5uZWDw$^t3@2&6=%M2S-CmYM3JE-R_BDx1Xv27v$>DG-5<07xR)IH#ZX?7q&-IsFB@A22WT-'
        't)d6QD3rV<~Q?81On8hvuC$2DI(n6+}zyU+}zyEJdWdc>15m;=F@qSP18~I<NKHUQF)q9Ceb)AqWNhWEvBQixG1u'
        'DT0})U%Zqt0j^i6Q#zlS>4F==Id{Lx>L6n_gS(Hqt`8=6t`Lw)oLw-A*pH0-'
        '?zm)k@v>oP?NjjvOz2tbP0K80Qvut`I>WmlD;XKbLr7E2ii@`7%o~ELFzQnq!@bPph@#+m2+JkbwoTM7W+vjgyzk'
        'NS=^7!5J!P9r|Zrr%>4;pIgEV(SRU(>y<_Ki_Gjs~+yKA$WHzm%<Z)O{G0^J4c#1pfmhc$>n=r$nP{TFw&~%_w?9'
        '^`l}joo8ohl$OI}2A`3b*>nOlj?%N^bTk62iHrz~Fcuy4^6{8zNnB6TxkTX?ASYHXM6bM{)$YN>Op;;RiXYr8heb'
        'A<$DJsC@VCFIUu_sAj;BbGEEaKaa$2)wImwgJuA=?{477vvJoJYB9Py8mlanGnN#?2KiD;C}6PWyw{N2qjke`6QF'
        'zzUa5fN*R^5NnvozC^x)9ElDLB%~{jNWK*HY*i29Vji4xsr02WqaQzlQQi@WjafWWS$r0UMue4ICtZA+X2?1-'
        '|U;A&^OrRZ+~0#``dTowywuWs4LtW1}i)?71CgZzb&@=+jnIx`m?guJyYwRtR?<C&*qa9MuOAR6KvFiX;8E|h|kh'
        '_f<%uG4`QkuAGSNuKQHomy7#_Vq;1YFh8}}wub=+;<?~nX4I+bqToUhJyx)I57ywEHFod+OtCkN{cfN%izS)2Me*'
        'fpeKZ+)NWz}}{=JC(_uOC07MshM0oo?LFqX9}4z*7&t0uBSAkP_V<-gtojM3ZECvKOb*7(b-'
        'R=pnR!fb@%or%6$!^SyX6A9wG@reK<!rF-'
        '#tmR=w?#}SH1;M=|UA{)(5_eSY?HcY$pqXQhC&9h|Eg;|<_=n!Ctgb!WXKDa5qV+|5Z4{roWilsMYv8jlIzWK-Kt'
        'yo~?EFEU!YzU}<tSB=Sq_EJUc|MEsF^MH>Db9-'
        'Q3@V+cWiR^U%^Ql^1d0Uym7Il>A{%w*=^0GKJnbe$nv}ayJWr0xIEsl5mp~ArO?*GtrpPa#NwS#d;y-8E6y$2F--'
        '))yMO(C4Sr|yLHV)YJZX}CTPXMcXnr0`b^WErne|D+9&k_=<yU}@4w7NhY;G*tUJL+2>wu8NkwWj(8RK>*ZVrQ+}'
        '*h?q6b@%+Dtvbuk(_%cyFLtBTY&1%z>N{*{d7)51taXbqjG~J>%WkxF%V15PFXV9U^!w^Nw(Xi;V0yAynU$*e;uJ'
        ');7}9Pu%`b{%#>e$LHLtJ{FG@2oAe|i8_BJEkSyG%p(Ig$u#jFK~bGDen%CitmCfUh!H!2YQ5M7Z)%Fvu*!F?aAe'
        'rS^W4rFXgkTu{@Rs!EGkrLB%L$Fb#K5X^-zpv`5mke>`L|3LS9L;@ayt8DQPJ-'
        '>$EtA?6Lf|g);)CGfPYf8<tK+2bsX3mcmrbd8>n`JIXxBxS+>H!lMiDVf7x?xJa2tXwO$#kOu<jjiEq9Oe`8+?{j'
        'WqBu=9A<&o%r|*eXGldu>lTMOF4`}nW8ES{J{vdwS#M{qCpQT!3qvFyT_Ux2>EizIS;u!`bsE>$#h=PL-'
        '(Y}7qdoG2~F<DPNKwVo}Zjd((dsBCUSa^6j{=p6=_+fqrG?zN=kgFF+9!-P)fTN-'
        'H#`r$EXgDli>#()O5s)yJbE$Vz#Pv#aar9#(HUCum9lc0~{7-'
        '({k5Yd+O({ZQ4%+ig0j%Z5)(Y(R;gXQFDt4X^Zlp2&VJlX=rSyR!I~-%jkq`QDeD7W1-'
        '>kjf1eSShc8fxhlt6@p6YzqEcX^A_Oc3=ZIh@jR)6Ia8e%{x>~&&sSu)-'
        'qY6QWW7qjfCs@$?rR)m*nf~#Ics$96AMSq=)YQs^%CpO;%qQ7MtuOlCwySlEWRxvRwFbqn?%Akj8N(TG42v;HWW7'
        'N27(=Ls^|2~Qy%J<E?bWnngKYKM*R)-'
        'nfxfitjte*;NXc#15rahQK`CUTd~pnNaFUJZe)o0!8C<3)oPhxFzt*DLSO;H2IK;LKdOsju4vTz)^=dvEUD{$b_*'
        '_4ad|;w;+S6bkK~5|Qt2M}K?H)hcd+c|2_}09x>HJg=TphF!g>3j?sZjhCSY$+#q7~$P#^_MV^ziR79<>{DES|$)'
        'FiJ%#<VncaG^Y8SpJ`CV2ZZ2%?^Y-|wt~+OEL)}pDGz$;dbfkeDIz7R4YdrTCcArSPxmJfH|1G-'
        '4Px#LcN$h48G&Da23rrR*qtOL?A52)WJG|~kd)fnzi;hqL)Fq@Y+K6X(vrrCfPTjpo4^btPQGx?eZjeJfQlw#ZKa'
        'W6-`!{jCi#wG-rt$ESR;No=ZGSYr<PSBM6&$v?hfzm+#A;|#&v>+mrL@RE}IOeser15vb)>8NRK~c^RC9m(@C}i%'
        'G0(zubcMKofja|9OZq}HTxaYF*Xs9n<IO=I6Dprqg!`;HD}uyLj`j*8;j(#kH>!Z6nPtk1c$x@jB+Q@jB*Bb(7Y}'
        'c#a?-u%u-n4&~TREHU=A5hKIIpdJfe}X);l3Tb(Iku+VS0Msr3Te<xsogT09FGCpijO*T==-'
        'SZU4FgkSL@mim%?wNzg?Ypza2Yxnd{ubq9J|&%RjS;Y7=lnR69>>C7dPa{jeN~}$y=A(6&1K4Q9%oE`hg84t>An)'
        ')c><*x;O2TvBm3nz&vC6-'
        'nv`a96cfbi*HxxuUu9ZkY1yS)Kh+6R%Qk1ii9i34YV&RHmQ#Suhse=H!a(6KIqH9`)b5_-'
        'g42KKbmn5R)!V5AhepapGEuNCsKTNqfjJ)o$N3BoX|Oi3If#mWZ|jz6YDqBQbx_ckWM+1n+^%SnUV>7;C{Pn3Gop'
        'p2oJMZ~RpYaCUSz{A3M~BtrA@72b{hngl3T(UgnC2U(;h0JEABVyj<D1v)8)l!TBK{y>yFaNJaJC1Tl^4_%x2w0T'
        '+Zv|Abxjkb1OUHZ8s4?4OhwbrxTng^d;)EBAaTnkaKS!>;J=9I?9r$Wu07ae}`L*8gjsl*vtf|`Usjg!svLHRGi>'
        'D+vgXmqL00r*E`#N2YY_$r-iYwS2*U3Rn?H4YZ2aThwH8uT10-'
        '0$Un@I6KpGH{PAtPrXA5m_kDd{y?1tIkifg#4s1@NrG}w(T(0a#_L%fn!$ap;nW4_jtmv}VN^gPKi;?y~3rq1DkH'
        '`KL+4jaNb6n6MXDuWy`0c?2Fs`t?G#jo_3HtogB@87QVN;HZ9xe(8P!K!(w-'
        'go~ha~d$w&U=TzI0l9tpQpsU%}Go+Mt=W@IjeQ(`k6+uWwx;HfUDEt?dg?ml5ix##?mwpIuNwS_kOwA+f>5bJ*@w'
        't<O%Qg<E|g!kAZD6Uv&^;vvT17?W86M-'
        'lAyzTXF0G=uCJUMhUoy4OED3oi?0^D>{<a<;iW`;|Ux1I(o<r<<@*RtQ%$BnQfN)*e)i58dN*oEIR*aP)3jGA7$W'
        's0ieLHiBl!c@F!BT?2o0ZRpA5B3YKfDzcNB8dhI*GYIF-bT5{M?)b1%S!JA^fZ`PM*={l|mklP*RUbBBzP%V5lY0'
        'us%%NnJFbz>XT(^?>d*W}h?uk7v%n(g*sIRIeh3UPE4o+#o`sAx+o1Kl-Jp$T<?e8_G<|z7QQO>jRQW+U{BOz9f)'
        'A>c3vUMSDFz;}<($U1b)@Nr<Rkr4YTB}D|n&dsxNj0^)XGt~<$v%Ef9=2jT)H+Ve6pvdj9r{hspaaVEgdtL;Mc26'
        '+*@|?Z6M2m)U879bQm+)Uk>TAGyIU4JTFy{A{cEE|9(-'
        '59R>At>oL{wM`V~v2U%6yx%Sg>Jy!oP$D}*kOY~@v5_)yWEz{;V5&$(>s<iEM`Z&E$r3eefGRbb26LA&631$phJR'
        '0XHWz}C1RIQGsUDh)gmZr_39ar+bCwT`6J(OQC4)pGS%&#l2Ja=aRXbzt2J2Nw~sP*hZcF4opTncszq{$Uce&ETl'
        'H;qoImZCy9+^>`g^&+<_!?zQA-'
        'n;mal`$MPFFyXi>)K%oJ5vpFD>w|_>ebv|G_{*r}4)|JU|7zcpCMOo|A^^voVXzW_PV2_BmP42(vX&96wQO@9s;~'
        '{%hJawDd67&@P?XPV>JZJd37KN+j1dG&Qm+D7^(6I-Gac$BXvx)^HFw6@fXN&A?Nr-'
        '&oc?{XqjqLgNsV63^?A*b3A+V+0$!?i6sS}+n+!#}3#@e*Xgp{Jac$KRw^2Cm2thJ9t}oj=mHKIMmP}kq%q351UJ'
        '<xS4Vs*F4TH`L@5~rHy*EGKH*#|FWLGn7PFyUMT{yjDGqO+{C6?Qe`EA5i^;T&Nf7a>?xi!@vDJzdb&P?9gaj7M)'
        '=$X<aw>IW@rS<UjjWl-~DB4-NL2JFO0*I0gz2eFNW;IRbt=k>v@}cb<Nq79!+6;%9b#qOCuyCd*uI=N;#!%$p*_-'
        'zb00p^Ai10fVtaY!Bvpj;p@`pumydb->x~-I_LgnnQcOT=<v91De!F=fzv-ML6g4eO2G|(egJWlPwO%Yo60IyCcY'
        ';BaCN5e@{mU}VY5QrlOzbc7EOxpu#{a|%Lqm_j|91)2ogb>OuplBRXXaU_!?!_-'
        'bCB}{@NQg}E8Wunku?P;pNEpOp3@8S$q7Lgsb|B(4bnrqXgd(a0%;@;=wfyVBO#x?J=goT^V+gF>t?`M7aZ4dL)V'
        ';X76+e8R&!RDi0+7<>7ePhZ_hAP~OBb29tnACs#oXU`$h=@>Uj!Mx>Rn{s6zM4Yg18J9#koH}`+1R0e~oU!-'
        'jIDMq8Q2JV)KlOA}n%Id2kcNL6vAw^Ud(4y+u|D_%R@7CZldXoh-eWb`WjQ*r!1T;~`Dx@lZ^t1fPM7?J82(gyG`'
        'p4c&r^E@BaB?R+#{;sDN1R0k@w3p<KQvOURxVq}fH4gV+m5dvJwC=LstksBc3|ME_B@26X*TifS%;)f4z;^&9FaO'
        '>`mw|}~Kx^?IL-f4GRSE&THbsJ%Q_v4*k1K_s4i?+5-d;P5t9H8>~2_tg#e2X0>Egp(Y2z>J8H8buTk$v;5m=b0-'
        'b7w+CQmY7Kv*8fW;4E|3i`d~J@$sR}B&O-'
        'C$bp5=(7k&WyJ*tK>ZbVx2EuLL5RHY`X@@qX$+cQqfyQ<<IdR@p@vCu0a(RWIE157~9$c&UY3y9@xJK?!B_(F)u%'
        'gx%fEeeKERRg9>U+be@-~#0NSuSf8AJPH7i?@Vb0*c~m(%TsK44e_+wLlS_!m+*(-'
        'A0a$;mXw(3Dc=Zz&hE*(8I)a+nvGECsow7rlp~A?7S8qEQOlAsiOc5)QH{w2=)HI9jApRz@f3bOFU6BJ;(Zk_%*K'
        'X%CjNy^gIRxIcX=keSdUf~sU4DuJ`SB05LfjL?;hkP)$W2Zs=w3HB9VU0Hd0>!DC3C0L;SEvJe<Dm+2pc6vdCJ<m'
        '`xqr51mFh}rBSO?(bFp7T2rWweiG&)W(x}X=mJ}zM(P;xVx39SJBo~9tvwj$UavmxwFWi(Sv(Ve8{fE~<Yc3e=(o'
        't`)1AdV&`MbJ;tn!d*YvfY#vs+ug{Ul!EQQ{bF&1{@?!(<R$rIiup<+lzjJQH-'
        '#t%ohca4A)sFI?X2|<hfB=W+ziP($aj)7R4+t1zC?5C30?wy#TL3KX6DEH}{}ad_aa5`+cvDLD>i!Nu+>JjUXjg$'
        'Ja|GuA*0p4DK*xFBjUzB#%CjvE()qT_Ttk>MZD-'
        'gf;yIe+WR<LSbr?y%o?^c{(1aXpO+RbJZliKX6PyvgD)Yn@m`QI6&#d8auR-AB7t9@Eyv(rQ{r_0N;SU<~gjR*#b'
        '=`Y6c<$x7VO+d7ySX*Gv&tFBUgYJFpd5uJdGQi>fSe!7ZtRvpl_tT?YD>vbNae<*`Q%`dTG-uP@qd2C5hC-iHJYq'
        'FOuG#(vlOZ!3jBxvu62Q>AW(6juf2C|Ja2N%y5|Ruj&?;IJuYM~Fh1ePFYS!`6<l!JyHM+BYN42v9|Fcu4FJZ<Fb'
        'VQui34AU`3^Lx>8axm;v$kVuo^saVB-C^fNq(Th21DAP0}y=9mz%4C9@EX>`oAmbSP9u+|7eCjOmnTIDIk`?D|-'
        'A{O&S+#G5c=9GO<>%gpY*Wnojfk6PuDubx2QAXthcp&T#pT!Ci5zIKt&bXRA{7wQ(YQH)s(qHhyY4>AObQIzFpJH'
        'ZpJh-9nicGXDi5FL)3My-'
        '$Od+vOakVdiv5jKsPWdyR9f<>5XynsYhf`apb3JGI}&t?l2I~4k1N=P%hUWqsS~{hXno#om#>@aZM+pLscJYV<8c'
        '1aAVpHKmf&l|@W<`uoXDSPH|GoH%k1WQI+nGYxi#<C-'
        'p*NPzk%Hxq_Nq>YudX)Wt5e;pN$?qmt*(>x`4|aLP2;%;Q~^0M0}2fN+7KisrI_<;3-'
        'f_zwR*}5Apw1gdY}S*fRySB_%Zn5tbG|?7xiQxD8XWkyhYgZ8+?=qzl+LA?tMkemn9h52Y-'
        'i=DNG3_`(pZ>;WxW*Y4maq-YgnVcu)Hsavs2pL<uwNlNEQHYv3PMnZRI+3-VJc-'
        '$cDN4iw*F*eUihJUz^RpC^iephb?tVa;?(?wjjo~+tF&$$$J1QoKb7(*f@@F*Q8i^&{dyc4shK7@$1A?yUVPze_P'
        '{sa*5T0MUFEG^dp4TzH(Lgq8M*uxX9iW>?k$x;;t9tM2_t-'
        'R8{!oXci0c8M!3TWghpi*e_meq==wVXe%0f?b<Jx#Pij&G(kMpaRiYi&br&}Kt*aszZ&Smq|!h-'
        'RP7V8yq&FQen~{jk<z#aZgwO74rm3j2@Ka=)~W%BNQ-shewz9L8@`WwiZO8r-'
        'gGAC`zWy^i|m>}kMvCuqbgeZ(ZbwrhpP!?}eG`LG0CzG!gR7?lZf&x0MB^%~xxYqop=OL^+W>Pk`m4k~fptXI-bI'
        '1UGB!(9tu{^~XWOLd@2L64vLZID%6`q0prO5bwUOeM7tw~7P0m2%mvXsGV#A3mN;#9m*ev~I4rHd9_}Si!1Vtxz_'
        'HP@_NwHOA&zV}P`EYGbvI8d4vFbvIf0MKV1}E0@0H{YcdEFlsCXs6A*TfJBkNR6P}cMECCgzEV%Go;nF_yuexieu'
        '3%@eyNZr0s2&5$zB-$4=$)YJNEVDisgk&vLt*|6zAZ#={PM&>*fb&4Wm%Sn;*_~7f-bVceoGZ;C-'
        '!5>iJOj;_p5$Wl?DF*?ji_?)V_?$fC3CMEG}HgY5(`I{0y4(`X_O^dHZC#;aopns}v8zy|I?&N-'
        '@nccV@9V@_HY<ZGR@8mK=T9&^I+y0-'
        'Q#3MT1EIvvTQiBBm}tWO}dha{`v=Q<zR8_SnDAhG$WJ0Dr7UHg2*mBrt{0V$}ve#T4v$C2KRen=K&nSt2&)3fK%q'
        'D)KMA5nX5i3Yvse_JHec{U+e01QY==EAx!%RlzQ-kRP|dhTKdB0D0d-'
        'g@<T4pKNN=h6HkR}O#3G(v49&%h%6=S2o3+)#C99UyueOHwzgD72k#EJQWXj}@Tip!AupIqe*|pFzUu4!)3dvk#Q'
        'D63%evb)}ouasJ~a+e`Nd{TZcNfI-piNUJ^|&6d7E`}4F+#!4-'
        'MxymotvuqfinVl`pXr1S0!aA)SSJ8Kt9!!LP@tsCkr`Nce)YJXjv@)qkr1kbC8J&cexfGui$)+D|UaGTue1z3wz0'
        'HFLP;cz8OdY)bCN_>uEgb*Z?HkRk8=RW;W{vQ23EYU)ovj1MNP~eb%%9?Sn$&9>FR?M4_-'
        'M(fz`l*GmI4SZw)o)>;h4bCsP)C57@&O9tJ>$AWMWRi5fPksJ~X)~{4DovLZVzxeR0Lpf6Td0P<)dEpJ?&9^dQON'
        'Uq%V?c8EGPNY3@g9Px(un(8$9?HuwLLSI55V)Py^&K4rEt0X<1-trmc5=m)MM<fO)QG{&AmVHOIA`t-Jpr{v-'
        '7%U0Aj*^;ZX*an@FugfyQM0rlH&dESaw8Sz%VdthMwHV;b&Fo&B~(d@8E9Al6SS)Iaz>%a6ekDrqX4AGc-'
        '(@Sq&PQNifQ@Lhfc8wetQn$R`B(D1&jq`U)v?@Eiz{`U7V%#e79x0rSucJ=-Q53q6=7zIjx0IC#>)q-'
        'F!;*@g1=2^^`{%NORqhRsa80$B$JV88JVX=BO@G(5uyFU~_}=@PWV}qURvgkYB-nAKVlrZVmAu2h2YGYid=pe3hB'
        'Y%Nq0PEL~&r*5aw3j8fEVg3ssm%i=v<OIZwy(l4kd66%|%iGr%1NlO&#>B}gI&lpqHUIcLNzP6IcI{S~;5T9g5Py'
        'JH9Qi5-QFQv}-1l2se))ZE-{6Yq>8<pTD-@AK}5L2WD3Zr*H0q`g;hRI||(Yqt`<cA%sC5&#-'
        '|NJ*0?Aid#d~V<8*<DaeC?gmn^3ePgL&>3P*^5Z#ViY29`}8C&RR$~(qrSNTt|H%9)sE3jv5dS_eylU8dYW`PeR~'
        'PKqXq<K6kyQkb7-'
        'nonMn&$p*_x(oYFs}dMbZDd>|6N;jShBARS7oQl=woJC}w~dPL9e#x8Sv^CGVvsEm6i4UkMCz9^;{#yIMneBtzFe'
        '%AL8kh)Ypd}QHN_ErKUkrt0jFP6Z@P%)e`sWRESCG_kfH(QLw78SrTWxE9uKREehRQ4bGC3Cj>kYvHeL?y%+0y3F'
        'J#hBdM)DZ}>V?fZxTEG#kuQ-'
        'E#;gNkde>fPt*?;|h|L4I!z8egnsrYItNI{<9t?7`?_cwnb*A|w42N8(@(%a{6UcY@mc=Pz@{nwA51;OE;tyVkQd'
        'nof9#&}%Eof-'
        '~5B3&V6(ukfte*ZYyi~jN5>sLKG9=Aq09QE<a@g$u;pU|RtvV1XWSy;4tXx)1%bg(uD`05>ii%;kC+3wAo7Z(@3i'
        '(9?CIJvpq@AokWl-2vw*ZZ&EVuOSDcl~>J?v0YT6UDz9ZEfG)9@C#&$#>&h-'
        '_f6Uw|Dv@@%4CoC+Tm~pZ%Skev;Ck!}0d^w)k^v>&~t1d+{Of?D=2cyngrRx6cQU|N7$H;K%1Le)#b{blvaWy2Y!'
        '#eErk&m(O3lAH4tZ`P-'
        'L|_XqEuKHh(hII1M6{a(M1oopZXlUpP9bV|RsO$B#uZ*A?2t%BRzg!j(Yc;~KFaOXA_q~o0g@nr>fsl|AFyaQw61'
        '@~@?g7mxO7Ax42SdGWy(Re35r0*61Uc7q${O#kX?_a!r^}pWW_Fw$z`TmO^U%!4f*nj-w`95-'
        '$e&|qHe0HPwO`eswuoUPLl#_x|R>waiup8@wV~{6<VP4Lm<o!hor<pVs`7n96xT*(FU%&eP#Sg5*A}wc73>w;Qgz'
        '~3jzk@o-Px9S}lI3-'
        '*M~43?$Mm*V0jfk#lHv*Ja^?H{ZHimjF02@Fn%f>B!=rr6#h)VR?A<K;K*5rwB$kCW^G$y@$6d~#PAb?gXlv%R28'
        '^<MvV18E(u?S2GHdBasVG6+65~(<g~I==vUr$zMLx1$19sU6MOl?-'
        '%0`w58fVjt#v~q4gofgeulDw6T+uh*urCYHpr8e)nAYbw#^0|<*7vgVdxF^#zgt50v^o{L0(2@0J22NL*}Q}gw@C'
        '~Oq-4wJQDi~t&GY>nl-=~*94(nGjZ-JOI!i9`0{MG6U(X2at^~hoN5Gzc|JVO!iPc$JpgU^_nXMo)M_*szgS*imZ'
        'L|=t@3YHv)DrF6tKT1m`Ws`NSDO$(&i^K45clX|GIXLfRFeHw;r^zYV7ER8!RUP!08TFr>pj}6PSEM06!+$yFrM`'
        '2+v{-)Tj(6m**I#6$`4^5YlGMU{bwq7)dNxTd-'
        '=0BNv9|C)5H6EWzM71q&&qP&^*mqL@kNoX@aDL3C+iZW?R*6wdjur52BrR)D>U9HT8S=*Hb|H@x0Y<x1)bC;BE9?'
        'CiBxC9CQg>y8=&35Ts(sBM$TAq=R<=1y0e`szYxcMPhlkeTq&2A1KxGV_2jh^`r<vVC?`24nCyIj*tk4NQKwOzks'
        '~$p?i3imM!WX28ghGWjcR67k+1H%e1p*icnNCNVa2I6yS%y9AaUu04YF`dj5H+@#<ac)jNRjpN}T^qOUY215L#4D'
        'EkBfag~7J-'
        ';;Z5q*B&OBCUvS7O6xHSa;L3ARisv5;*D!A}nix+MZ5FPfxSSs3p0y6$`T~Y{^%I1Bija02&PjG0aeGO!IL&L}{@'
        'T0g1%`HtK4nNNv$tkD2d-LU5sc8BP`>m_k(xR_?#hAhDJq7w(K|80~hOWl@*-$vsy$v2&wa6Gp>-'
        'L)LekF=;jLMAG8kiKKbNOkzHLN+z2CQ}Y=GUE*CjmqTmcx9T0|muV?0k-'
        '}jYFJ>d554EN_zU74+SS;)g^*REvSSmH6>;eCQ{A($ht`O>hd=p}u<>r?ob9kKqN&!fumXO64W|Vnk@eeE5^<nfa'
        '`wp~xl?NIIn{GL#t4E^BLI2Px*u@QzH(w#=etl)(zmnY^@dW!uvS4$3_xF+n3r$lS|A7r91r`}Go-'
        '!n5Z|#iLl(jXRP#wbntH3g&*<!2Aon!F~_Fklynz+`3l}0uRSk!?_q<Ce$#U|N~L&>JI#T=pFu^)iY7@<xqd0k9K'
        'Cz4}_8Wv2V4hq<orkY*aOGcyTm<44&D}lLyEn{KqgbrE`GYi@=O1BN60gNOx;EomqRRCj7=czjhkz7A5_Ko9B#FH'
        'Wnd+ZpBpfLpr0<udk79|HPkw*}`N71C{=)ozkSc*L$FIs;%sJe7K{6jlLp$XIs5J;`>ym+1rPyGe1i3B*i<2R$<1'
        'V_5#+YCvK&2W+?MXMGSSweeZ+u2;@n%$KhG?s0vkd|_qU%VG4R+OkIj-(Tzfe*C%4yJo0V-'
        '*ZJAkamluUpzkt1*Qi4i#fz-'
        'L%b8&{}p?*AUsxZN?DeP3Z}Yp1Lf6jgTIKiv22+bc|p?CX$j1M;9@6n41j#`77wlTic!J0yOmeLZ}#j!Kax(okk$'
        '}cnnKheYx9S%`T7pPUiWnrjMTkef%iyPIxE!6Ys^E!_#b(vN;qJWlf7u7h>Q02!{ZT7gKQ&jIfkHw}ja;A{SsfR{'
        'C~NJ_{^1VTqjsRJ~^80-'
        'oh$<!D#2mD`uC2b?5WZR@uE6>S}81pR?Z@A9*hoO5{`l%1;yZ^LYynAc3PzL>sX^`D*UY6pSiFsd6kdnf7q3H}D='
        'eoAfL0zbDbb&nAGRyOO=(_hiUAkVQ;@eOdzh`KiDF2{JqReTtzMngQB<KUs0umX{KQgt2JK}4$8W^&nL-'
        'F9fy%~qV-'
        '5{mgEa@Z7a>Rc8zZ&%XLJCXMKcqMj!eW&<}PT)2t&+s}eUm#0pK)u1>%<kmJWICFpMafx1wRF@S38EkiqqFTuvSS'
        'HsJ_<mOeYRDe2C!hC3Ie0=j%J5y5@=~IU<Fg0OBAi+wx!)vH3CHF8Pb6e01OfOV5v20VJxN|6#0p(LJn7w+(^p#8'
        '!>Qs7`2x@qcAFv=d?rdjo$`vCD-R}WbHV>OJcp)-'
        '+x}&Wd5pJ2Dd8(EknRpwdJWwkV0=D!HvLfq8bE7%m||@2;M0l_26vcs@}szInU3aCe?Y=6TeaIat8mnD3WC_E9u{'
        'ssMfZ5sTOggMfp?Sc>6IG=M%DFql+i8H-(B@hrlBfh7aNe&RaCX-sLL#_y56!hgG;-9a-'
        'naU{s!q3_d2Y2A6|F3+#b&LOh5KuEsSCmJ72Tw<P+V=yqFQ{TKi&A;204OM+klU|c~jW6T!iY3qiCAAAOA!j3OzA'
        'GoO#^=)J<L>AL*?zh?sv=YW#I0GQ#!O`suqhN<+?;KswQhO8<%r=0z<$!_RrykOZJfPE~j%C?aAa4agI-'
        '5=ak=uvflpNT5Q?S`ipcyCLO|##{K>&9S?TPdj2|Y5&j@6^=+&-'
        'FH4k6@@wrYpX4|K5PrcGoMd+}p?b$GRy%R9tYARuf9aKaU**v_?p5Qecf@i7?fkk~=>rAR?DAEtDI26DF=(o4L3g'
        'yv-Yqun6_uuc%PR?z7%>CO*sj!%@{YWC1I+N-'
        'nmX&EoJf^WW2w=wLiQA_KJe{NeMi&d~>7vE9;<zLvAYKGI3+BN$k)Zm-ZHrs1iu~38A2dS<?gO&fRbk$*|Vt?hDQ'
        '{zomPKIw{aCE(Zir1jE*6u<|ooSy>JvGeMnu1l0w&f~W4K!iYlGvbZ6lM1s@G7j$!zvo3#d~z4<669Yp0(o$54oe'
        '0KOU1-!V8bBKmKqJ6I4<m!k2}4Yw{osR1zyjS7W*bzuzNFBhsk}=eBGczF?T2O52>M-'
        '4g;#1IDISLjkLOUkutS@@(>>rG+zo2%-'
        '&i4asIn;L3ro8@h8`(AO_}YP|UJc`^r?Vf}6a9tZNvhje*J^Nu(2^8VT>jvhS{SHYIXIIkl4NRr&S%HhHt>14H@7'
        'oKg@p8;$IQ^U_R_WnRk+I5`&l3clQnARy4#O|5L_jp~0i|#<{U{p=pQ(O46BW}GNqo%&#7FUB(ZL8eAgx9(7_(kC'
        'mGe^J}Dv&GSs_ktwqG6pPAs{27mI5KZ2Xul4fVyZm#oZNu^n8t=2?(tV5DdX6o5C2SlX;?H2rD&idvV+jDPC+I1e'
        'Ab%xtg<jy&eImU@mzr_C3CV&H0KR#Re@qr>nU*c3BJqvu3v3b4Khu&h1IpHJYULIyY-vt3gs_Urf<6*3G)Iwbevl'
        '|F%NcCX$7iA}NgG6;gz0vqCH&d{25v%32jIHBjEWrR`u<jms=FL<$&KiFs`|;2zmStz_D#kQdH8Ws1HDF3U)l1-'
        'ezlKYC}$tWFe}R?9jz<Iy-&IhN}4u2^5V=>6Nl@*<fJ(YeO>b-'
        '?tUpHzYt4dRY^9u~^w`>=Pb<sQ3!rsmd6@t)Cd5W8lQ<Z^2_>UW|g{Bye#U2g9}U+{Aqf3j-'
        'Gav#)Kg_jeEl3Ir(-'
        'aHoVyQE74=C?wHqXbX5MY4=#KlnjNEOzP0MoaUcR*o2Yr~0^M(b<yG?j^pwF(dFB3PH(z>o_?XF8}(o&&%3HLb|P'
        '7t?km0cIWsE<Ga*e0P>Cxhl;FELtf%}O8nI;^Ru)i&Q`W#$yW@VoYk;pIVc&N6=X#gqNB{KTsd^o%*Y!_3J*%oVX'
        'q?fCidsWm1J_Kpi|mjbnkW_ZzAakwioSe_iX}!csj}OSbK#Bz1`@$JD}TPW$`B*=790<;>G)XjxM|Esv6(!Jjr2K'
        'OeUZ#-s^)d_}#X1SYvlDdy%?45gm$SrT#@d1{(jU@YYuzvH>14^$&rY-5j5p4oVp<M~+aAe0@cDt>6#Q#p<X-'
        'Au}*R+1ZH}F~Xo>hA)hkxD=mKAfda<D)PAcATUvYf}DxiMoSDTTs9Qf*%~~nE8I77R3o*PRU(4)^OA(VP}b3ziI~'
        '~oXq=+tafjhAG$7j>!8h=;(|P$ld3@TdOKdtta`7|YbXsOGmsPA~VJTcg8w^p1;6ps3NK2cjq`n(*yVNxhJ61q7w'
        'Xka|sUfZL+EZ4Va=AT5+Gg7hPb8BG8C#?$l@=u9Nis(VEls^PzdpTuNy}3ahknBGO5cTpjMadN1R6nwnFB7lw7|h'
        'n4KBId1Y84Vu0X!q@9o@YmFiJq93Wk`x)A{A;D6eoi&)X?-'
        'X1{)bOah*jDXp>FS`3y{;>IXnEfPVZg_$eOCJD{1V0Tk@iH<n;Q=sJ#0vgQdb%h$)Z8O?6{4(QJBLY@8iRv%Q?V2'
        'sDx@tTBRNsgR6QsKD_eFTmbK)3zglRHm){rp**~R4?f~gq47RkVd>P<Z+1hDlF~Ksxmu@rW*8cOeoZj(u`t_n7Eb'
        'vw-AMjG~&r<M@Ypo^ol*UZEcVn}5^haQ<{#|6wK5(ZX2e2MAx-SO#t@xq%R=AsyK;}cK>b6BNx{>K@YuohVO%(`G'
        '+?p$UxA)9T9!kPD>Tjc#a59CSFRAC7h5(lWBmCZpw$kp_b{jD>yT7wDF9W>@qp|HIVY*zpT{0qgU5fUO*%%s%ZD}'
        'SYuv|p)G(>;%o$2W7E1|?cisJnkZx3I_tFN!hOD*qp0K|5C)q#?w7W=&9sNMSoKDXisZDGpe`uwEwM1uZVsHVVv$'
        'n5M(aSBnYqjJrp+TCP2Jk1L{{+z)<NL*z2dP?kWNBS8#!#3&@Ukth$EFbl^(3Z_Fp)u<@wIW5Lhf%-'
        'nbF+wJM>)s91f-'
        '(TQ!oFbltC^OBpk24f~qx0&Rc&BhDh3%<a*#ed<Tkr#KQ_oe4G>|v50zL1BOd?$5arqxk7<bi}hHxJ6vDBS+o6hF'
        't!O-{@rcMzA8y!%#EPX2%3ct`o8b48M9HY^rxVLbw2>@@+W0d_Ek4mkJMYXX!8K174&H)`vD903H1e()HCc%`5z-'
        'aejuj?X+x=7iUsCR5=#-'
        '*y}*#Xuo68<VrkaMFpXbFEazR_S{=%Ec^$RD)lh63uI*whcMhMsRAa4dD#lPcxX%<Vn=3B0PHbl1*PNyHTr>LG8P'
        'N=`r$&i#Yrtz2Z;L=&VoA-'
        '(Kc^c~;f{LrlH|JG>Y#rJpi!__4Q~}4!3U==OG`W2w>As}rae>Gwg6C$qWu64y=30%Znf9q@lQw1c^WrD<mr*mYw'
        '8sOkLM~~e3Hwo{3}q<))f4GzaAxJgG0j`K@`Tf#%v{wbgJccb<4nzv6VeucBCGwXTji9ebDcSGso6pNT4qb6&kGr'
        '1)9_O8Z=t6u|uajV*JZS18uB^DCqY{)-'
        'B;Lrisk6;RoaGA)SMa_%<U&Z=CSep1lRt>O$RU>uzT=nIqj48tF})XkH}K60baWqoX3nfbmxMp0*NE|7a@OsQXp>'
        'h$&>Me4oaPirI4|>)X}qB^(Swyo!e>6-V2@FaLyXb4ULD?YF-'
        '2rS5#`*zm1)rVuc0Mmw%jg=`Lrt{vUjoO79+u=ux1et7IG{hAtf&Dh5_I&SsXu(Hn~m|V;-'
        'xD1`<9KC9A_%?C&fwG5kI_l~?@Q<X|rajDK<>|l4wA8m(MEw=cpnS$wH$D<=v-'
        'CV9n@WFI7JdR;+^*J>Z*w%2x9sr4(|kb=!Qu0WwzPx}4!6-'
        'F`rZBY0>pGwBS(}IR~Q$|1&y#O&>>6)^8Yak8uqrVbG{iZl;F9iZTYWVO8Lbs5L|v$gSlU^`>ZoTAE*%@Vic+J#1'
        ')@V9{BFWGU-SNFa{<2BzPEYZR5TmA_&<>-'
        '*H;RzE_Am(mVxb?fO0Q+X(aB_+l`+?AFOdIkrbaI7XKF!9o63Uu2RHpUX)$Ok2?9_IK^o-'
        '~aXhJ|a&<`&@`(RCLg$aoREVvEHWvExl3Sx*JJ_TMoN+tHx2o)?BlfV80s)u1edDzP;jvZf?Z6G|EEUOl&bsw%43'
        'dKZ%00y<JCYPYGH&1G=HZPmRsp_L&klw=XK)6_i=eTtQJo-~WJa*5&n-cW=*nsZ1v#eYLptU)>xlqrdr0<vh?_-'
        'iC$Fq*4+Q#651E)CbBI&ZiC?SlU-xva<tGE)VN#kI&1t8~sD60cO|5fe5fU4!r`CQ<Z7UJIqg93eNbc9!X8uR@QL'
        'O{S7Xyl!G!sQGBRTzw#0->xnb5d(|^?zoI#%i{D#M&8qgedoyE02l4q?sO#_-{KgLC2mCVtsHq-wDYPH#KxjIbLZ'
        'zV|j4T;u-HX$##B|=y!E?jzKyL1MU7F*r`1l!nmE&@{-'
        'DtEc=S#qL=NP5FzLIwqVg=A%COEy$#<cmRgl|6D)PTf=CZ*v(m0*~WM!@qrXB7G}ne0j-'
        '?2&1;#^|KwF51;U+N^F#i7Jyivjz#0=VK`t!uCWFILpYc9B0H=-'
        '8mjWQCV$Q*tE5(Lyk{Mo0E<P0SmO&`RQCFx@^~&ob{ch#I8#*y?RQC%F0sl1gK3~p0ZQm#KyRpCqqgn7^=Mmm4KQ'
        'n-_rk;SG{eI7&i;D1QC~15XjMcrK#S6<Ion$0x=wC`Jzl^{D%&}WZ4{4DhKTr*P#S3P88oxn*tGG&B`D&9i2z3tv'
        'MztmW1UZXy5nNBLJwf(E4&kubQMt6WlLn>&@bLlK*)VmPOca5?Xmoa><plikOgMR&7UEQK$Ty)l3-'
        'V*EYG9EqqtwZ{)dpk{3(r1=q@xLo-'
        '2%ioP7`m@b7N?aEz9^>sfqmm|s~hmf3uk>@$uF`lEf#$SXLG|!qfi(;1!t)as#DABfN_FymC>i5a!C7Mt3Hv!4Hs'
        'e~P7#gKxFICaHGa;~&xB@lPHYs=$eH@by}d`a~qNp-'
        '65^_5sDB2PQWyh!{hN`>G<2SGvQj!sJsvThYz7aQ0W2w}^tphJX1kY)J7o<3Gf2{d)WgF!SoOF|Iq)5t=Bg(nkgd'
        'kD(7X6?uMa8Z6X?2adkVy)~It=C+Tt!%GhHTwuzc<tNSM=+-fenpYHeb=*1QQ2Ra3+>jO{(YXasp`-'
        'izVvT$5YE;dft%~m5__q?i(v{~T8-=5X#0-'
        'T<W;d+U#^$@cJEfFiFf?<F1idtw{DYGrDnytCXg?URJqDWrWH(PKa?G8b)(K*9FVJWmOnX8TBqryR=wOv4|T00gY'
        'uv%kwYaMZ!2f{R{T4(@wg$CMOx;QbILzQkocMU)(vCI6@}$&k^%i<;mc^Ur~<c{A1qy}I&-jKt6d3dmW-ZHM=juj'
        'xQ|(ulzs4yo3iVONPr4PQKm0o`V`8{rU)W_>keoKQFXO#6sqg1-'
        '9`*dHA(bU`=~CKVZ$hfFCC(QRSe%@UWIn_hC+>dvqf>jVtgyI+%$`?|D3T65*1tTL6?2xY4q6-1g6-'
        'PnMZ%8(xczKcL6STocSBO(V>KwX`7#$+WnB1-J{;}(MH~MK3wh{7xt~Ibe8)Qdb%-'
        '>yv;q|Sk43@nD7wh+mFZtY4@m|<s?voSwd%XjBaI(F$HB^W8>_LlA4mj^X4e5cbi<r^G26-9d6(;7wxhV-ocU;EU'
        'f~(0)&Aqw{0RM#fP-0!|ICmb6kfgFSl=lB-`HF>O}bFK)?6h-'
        '8S$&QfJrH{N3>0y`<ldwb5W}wE~PyVsgI2QB%Kr#RVJ#A8rrCFYMfu4HP#KaoRb{^Z6+~soC${sq9%C!)kku=Q&<'
        'FFXK8G61#dpLE_y3;#EPWJPPVf6v`CafYnyJ(eJO8yPe*x@#+C6BLzMaBmO<&z<;IJKY@N$6e8%qLf?>&Db%PNA9'
        '1_#DQ-'
        '=SnK~>?8U*!3Qqj9~(pgnpu6~9)8ZOkVLxpR}mDz5!43m;e$)rf!op21?G*EJV%2m_vM3QHvT|lPq$fn~Q`}vSz+'
        '_B;jT47b3K|Im<bl4L;H^yG9WLT_I;Ps`nu*gcZ@YETg-`l#?cENwt*w-ZzGAUhOlB6vwcvdl~G#99!OWOkC-'
        'P$7yv1dpOy^WEj(JsiV^L?+yG54%52%DzyTd|FZ{aZ|I+utkAX`1hd&h90K?e7p(7+zV-pmzq__Oji(6Uh@F9_;u'
        'T4~BsM3i{*|^a%^?kGgv&FSy<P%E=9`N^NYU1l$!b^T7XjJE$|5t=<lBd71qRRrrKFif~x{mA|VQBkg9J7;opU9C'
        'G*BaIbQ7yx(74u87j`$8PWTconZ692XCzsf?oh#SVsLbsHG4|1=%WPm)>by!^wrw_8=)TfmB9)~>xFeBu3|mlj1{'
        '43^HK)?$?25wj&&xjYj&MFNzFBC=}O@>TfmOTR;7*RjtAUF-UIsXX@E0qAGhbWz$HP<<CRNYj<fw$X7+lp#63v*T'
        '0)g>U!)1WzfnLaR(Jquch^^Q_E{C-SW+Zh{LYgkzgE{T)u=Lb4vzQ9n|<I*5|_UsBVmV~#R~T^8vWe$AjRZl{-'
        'h+7r<}W~p<_{%)o7r?8+^mU_b1XyA8;_jc}$nFg*TYCOv4@v7cJ<;Jqt`T0Gfm{s;DMP&qSgBbNgW+rma+;lLqmY'
        'd8xa)su+-JqWlS{0M&C?CT_+-2eyeTp@yMt4VPdYdg)l){3Z&N?mdjRCO{hk`Dp04KpY#S~f-'
        '1{A&OL|!c@1+mGBlEpk1epD8ZuY&Qo8-}=Ywj~PoXb{Nu4qMShUnlauCYSBQFpo@<WxkkOQ-VVg`yYQ8E{fFft|l'
        '_ugv>Ot8|`48iK;vRckiO-'
        'D&IK6bIu|jJ9u`dsEteE%7z{Z=~Sq+<9rIOt^|f#w^s}>Ez`mAiDnD6<(nFld|q2YLLVn**#uwQ>*9NtX}4UKbl`'
        'ddJoVvaGJHqBzt8dF;t%g&v}yF`7k>a&O{QfR&sDf=C5l^GiH?7_eS7=f_L!?Zw4X~g{Svdaj<a+WJLR<E04;{Og'
        'rw==4vEv{QrIIbF&CVJ%ZhaeWbSw}vO@2WKbKlRO>8Atl}o*$0*bgGs8WIy47uU1yQQlF!`E{dhGmzsSG*J@yK6k'
        'CNxFveD0HrwIN{LIds`OpO0jb3i4{c5krbCf`9awj4mlC6ZL3}o*NEauscD^aN#b>G*taw`Gb5>{b9$2&a5xq6V~'
        'a}<%2N~5>36G#uTCs=k?^FkFgD9fn38sOI_m%Xz3;Z$o)p?zg)oOBj9l1a4KHJ>3>0dVa$9#hj#3iZJ6v-'
        'ynv<;*>nO&1M5}H0kQ!X-bw<>R9T(<H5$)2V*N8G7>{75@C_vlFpoMRBn;G$!69REYdsww-'
        'W3;~6mSQX}1IH2M($MkcI?U-MuoMcbGR{d$0XN6VSLi;Z%TlVex;A_Zy>>*k4k#(-'
        'o~;ashm9grA$V7*|B|;vaLydg`4yKyx+7Vq!{3QDH+{YNX*z9%-'
        '(xbjG{Z?=rscd9_vHN%{wHof>e00lUexzw<+2rHSmAjZw_P_8i6{O83QmeED|o48-'
        'P<!vhnO|3MMt{8&G|>(gB{V`lDsMeJr}qoo8YTgUtdv=y)!rsB`2xvG`6bEp7!DlbLqm8N(nfh8_XUnjCYgJyBgO'
        '_5w`VRzdlFU&Yj;-A}}cfoQp9^jgRLI%(H7wx$vt;S>-'
        'W?O$4JmcjWm+Frx%>EG5v!t@MkeC5zt!87KozIv%HX=FTsWZRF_dE6rr&hlHqf1u9Xd`SZwm|M6o&SJkxEM5|ubg'
        'r<kvZRi2-'
        '#v)!e4eh&T9P9PMxN2ukIjq)1v=Tp?oFe%}UE0t<AiN0nGYjWi_BXm*Z+kBYvYZ}v)jNTzN~~CO>n&%I0R%aw8O>'
        'Sl4?FC$Sso90yK$Zrt#0=iEo9tby5GBd_mf638;TK??rs*;j8qz@$$U|yABohrW<tJ=?rw%2V@5q~4nzKDi`cyu1'
        '+8cWCwy>gBdHv}LmHw(?L42uG1Lg+xJeG;8b>tJsz}3nm0!Sg<#YDRvXR+XU_V3ncrpR)?w7JFo;RR>Vxz<2Pd_g'
        '@ewFW9{6#+Q)+a}H^ZiHl+FIXM2er~Fv{M4r(btkLV6&7x(5q1H%k|``p-'
        'l>eDd&%;*%=);zb|0#lp1n?zU_*>&_ewl133b3FsLMg`uXd?Q~-P8D-7~&&SaWS4&%0CZ6!@xz7-'
        '60JMvY{AIS&ZnyodDMj=jmu9Yb`;F@Y4I2M)Q+yYH4M|iro(5HfVesVHN^_jrDdfoOK1w6r2WoRcoCv|hbJs=`Oo'
        'eZ8L$6j|U{1j{*`K@z~Sq3-'
        '{l%fv*s2iuKkc||L$fFO2*rS5B6Hv|P?G)hU#Zs;McNEHtkFs^hW_R?vudh6Kt0T)l&v3aO;z>G}67I?J#i&(-'
        'kv<_<OaVV|+}Lj2gRqJ#1xwP?)ostAmDU(>(j*_!I>U!tPo;yWtDNIv$`W*{7f6Bb!ZET&LKN2>ST2|nOxvI+zjP'
        'x05N-'
        'H0_xe07kR9yjLe+aI^iGlch^zff7S*|Ln$C*+7^b8<Jf$a0>zc}{1GLLJ`4dISpU}oo(+C)pt`eUn7|N@7i&#Cvt'
        '!m(URZg-G=_EVN^U=PHl-Pgqr|0`GetiA<*<k<iljr;I4ou&NWM6uJk*1?8jrka{)9?t7%rjhg%62+jMI%^&xFzD'
        'LWwxi|CmfoJE(g>PDkUbS0w)I)<ciL}zIqjchb;z+68MHFHcTd|lsxSf>DwH<cuTyp0|A-'
        'p*w2rLio|2*Xc!mH6R|xI!DS`Lf<@`~^#d-'
        '$1ecThG~0<P87$^Q<7NIH=Nw~9kDOO<V9h*JbQgP+f>&h2@~~F|8m*R4+sGn7D;|GCon$)Dlhzf*_SZIh8qsw411'
        '!Q|XVTSL8Uo~<FRGy8-'
        'VTJ^XaM?I)^1neB~Wl`18Rh?3xwh3KQIGxkjwSomH5T&4trwoO5Wha?~=3GB$b){*jwsB?nUiya3PgdbdVQ;F}so'
        'X#-3)(mCWsOo_0k;!IsR5t5JDL3}$eUoa{3n>zEhHP-Iwtz$pRvC{ZZjNzAV4$YP^MYV7>0%@-'
        'I9g<Ovh6B$%^{YL^P8AKAm<w;7<{HQ67kg05hh|Mfy@Qcd^7=RE)1#7QG1237?yksW;o0OLj{nf;xeOj`>fF@ZsV'
        '{;Oxmy`s41bD!HoJPsXNs*q=uBp8Q4%;^Oiig2aN(pwAPkPx*`ypM@n|mWVuvmO!4NunNbH?k#cg_XMc&{Dh?c`i'
        '{+CJ#?61L<4bugW9T%Kb=^lE?qdT?iDx)|kyVw-_{L*Jse|Nh_C!@M1Uc_LU#fs>qe4L}a-'
        'F_^DBM!p7k!BhZT_{tG=Ss#z-'
        '>w3^Tb)d~Fs4n<7*88ZNcaa~LAa9|Id9He)wWQ#`hC}$9Mubrdq!aP&rJ?2CJH83o2>TVk@gczG6tB~&pc5YA<dh'
        '%a_d_LowdFONOvE7sm#wMY73;Gt?+e^C>pDsY4SbeL@z6xF*Z{??2YcFH!?^F%9110a*Zs;==Fv_1tD6*9M#UD%g'
        '?<*&wM`14x)%k9E#4HXcn;VlEVVfl$^8RQtexf~VY<Yq^+!GVg`~Nu?O;K5!b%=%|5E(9m})x2{6rSI4ss|hVw<e'
        '1p|%UcQAunxeWiqLOn40;O^Ieh(u9yDG!lE~lT~2jWHvyG`IqL~U3({FU4AcsmUKHi^|nEF8^aYJn5%Ji0_{e4eb'
        'xQoskHEQ@L_cn?o-^hoPeeX`=vKfv;C_<=(^#3UrcaySI?h)-=Gt5-2xihU^K=vir-'
        'HROUN*ex&A!#I6!afZ0-X)yB4En&Yfv-CQa0)Ic0U<$={xkwQeWcAy9l4w4%ViZL*;mHGxr*D+x3WYarfgGrG#>Q'
        '9YYSSONts9;fKbfHMekt$m;VY1M8iZ_^1VZ0Df4r@l>atd>e(TPaw>ioHTtS$5TY12-'
        't3HT#9C65cTw*fl!_tN96b47`0*CcKw&YlsJfWHK2HVqR&Oy=8_Z2agO;IPq<Ypi7i+jEdz;0Q4^123wVqlx}yL?'
        '&0k!p~N}fd>+VVv}0W{Uuk#vc4xIaBkC3<GlEJGdyIx4qBxtuN0dniRts<Y9A`y2e@ncKllF>K8_}CE$SGwfWH**'
        'R%FvV0eYo{jhLDR&mdD69->9hTaxtTcgZ1>Nhkq$|F!FedhYL^@2l-'
        'hxM<vmP@fe*5ds#{Uwj@MD&UQtgrbb|S*p_5QAxW1IwP20|<kB<=!;@6gu)TR<q<JMCC$*V{m5bLd+cgkg3NM$^@'
        'i3c#ZjvP^Y7<O*kwBw-nnv_yH85c`%0~EtChV6OPz@&*bO4wY*?BTt5_piPJ<E+Np=;P_ZeX0%H4O!n*vO|QQeO}'
        'meMl6ItpETwXo?Mg*UZKv&T}^cuO3()^VR&h=ksTS3BdL142%Y(I^{8rM^6!YB%zZ72af@Yj?-'
        'bXC{s+q>QD56i+)cS84p)S=ILGQshj}WknbGTI4~qf&34<H?ZR*C63nKsSSSwNQK0%hl2YDwD0luq=?`YLz>Z#|;'
        '}o{+VLFg{ofQazf%mGN>{i9%$FC4b3ZH2nFvA`!lBw+jNicQC-'
        ';1ief6k~m27tv}bD+EoQ>!28S9IUxJIG2<EXyGr5~v>S$d|AO0h+)EWi&rcBVkdFN@1-;o{`{k+8D~ZB{$3-'
        'GaW*#Hp`0b$y`aPVKxvs6YFX$*!#M6(C1S$6Z1VEpfFsl6P&9FjIt8zpq}tm(>8${Nd?B-'
        'NOEZWTCW*ArHz;iP8z452!f7>j)ZfJ3}zz{0b?T7QpHm;qVSRlJj`e%0)O~N@r9fn%ttdY%V%FitL>_}dc<0&nWr'
        'pTlZoAo;^?ZHiIlg2W-`Xo&TgqP>OcdkKtyPV-'
        '{iGpsD^aC$jB^`eJFuzb1RFIs?OPA1e5w=<z2t(^?FpUiw*t7ckydwC03^rH*6Lq&~CIY1qe;IfP5M5il$k75Eec'
        '+1Q#d4J(2$PXXgl_Eg6iMoscl*@><-E0*d=mwkkorlpn0hMtWOb{W9jxsq6^!4kuJ6GQ_tW=69L9d{E`H$BZ-JnT'
        'ISa%*Ig#BZV`q3qoFcDL!Jy-'
        '!>+0hl8)lEGL=CF*d9)i94Q<BQ!azF(~mf#yW#iN;%%;9;v5GSu`rf9yX<jBTAVs3Vw?e*jOHkk7#9~J2GBHUapo'
        '$1_?<v0oc2=VF-u_e9uHB+Z(DpQs@AzKC@bdBc$l(Bt6GVILf(8dYW)C4WeP0llZJY@}`yF4yhvebW)_SLFD6T41'
        'bLBvx<(MqLFQcYeMVareiIK<9B4ZL)3p154oFOt;u;PZmvt9*+g1R=v8<)BupsAi)##zY~ElBkqF70ueF5PVpc!P'
        '+H$1rquvtdNq&`=a^l(729LgK#UmM~@5C{L^&w@d0>}K-'
        '@qXGQR~{^&gR)tF<>KrtDVE<Spnf@vF`Je%a!zuW<AFBkX?xEh)QF#-i|IvAr0`0lYD7D?4x16G^ip0AiYXT-fes'
        'u6te~BP_?=q5{;EUCBhVrSujgzBtPY5x%W~7&o_<)Pqs1_7wSb6{kxA!)pV7CGl>4}1J^1fne$_cMw)O@x1(?}a%'
        'uerR@JDl59eg$8zp7FZ&tR3pLAjh~)8RZaED{lm94Vu)i=I9ODoN6Dq>U}B%&Bo8>I+d7hHmZdu{cJu!<U;H!99='
        ')f;b6afIP3+NmQQXciHJPc%U^Y8ySbOGpN!);#Y8{;ufAmZq&8-7>`et8@8~7)NR;U!?O>oXBg03@1?p4thv-'
        '$YtcQZH6u#9peK|tHIFd=iUV++tq))A6xV|KEqSYY+tNy{B1`l1EU-0JXy?Y#*8a%!<K9%-'
        '$+IL9ZZ&<nbjI)4Hbo<$Bv>_rirlqKt=!&kUmpY~v*rSU^HMtur@=LZ&bxb6IWDTw10AG5AK;`oN#|=uRe=tjW!E'
        '`@vejT2>Azw>*LTc`)2ei|hMiSBW%!1z-'
        'xhp5bMIOB0=I3P!lp9NMjJc+So>u(u<*##fNyJUI*%PA4jY643+j`!)s*m|1lf%Kh}<&6Gb50Nb}&jYssXqD@Ce?'
        'BdXcFYLY_Nsgpk)58UGL19*S{-'
        'zFefjseu(R%;^cUaD~xNhQr0#VnUvZ=ny;^2(=(Qh{vWjxergI5iB&6=YEMEhDbR9K#~f8%o>Q-'
        '(G*M^9smY*57;b*5r6mN<2TM+PlZ9qC3OSr%j^V${A*!eZ#Ga_8RwHMA1F_=DsEC0@!D@hZ+js%RikUL*3`6)(Bq'
        'ODD>Yi1jLH?!^0vv61P3T9KzB;d5)3atw2jAVurC;m6~DRjq{hJ<xcJnhQZ3gAuGY%5pWqx-'
        'I>L;?$KgwlvYmX;IWpxEjMarI=#R(@pokBQhO0i|pR&Q=)+n~}D}{hFt#fb+I~!<kyd(8#EgaC_LH!#J7(5P#%vO'
        'zo$bUHid5nq=of$t6=Wh+XhUTEZ7nz6`?#M^e(Ke?3xcpPPgrD?Z+Ckwjcc&q{B<*;ZbqO;p8Ow_k;e7Cq?_R&^m'
        '88aJ<7Mjrz8_#8!b{-'
        '*J32fRo<RqyKlpIN$&@I^Dbb`JDDl80`*2CRw!jT<$6IdS7|MFsZez9*H1@UEZ@TXU551>s12kD3I{dp7!N44vN;'
        'H5Bl1sqo(n%iW*I3#$ex-'
        '0ycb=$?64;j%3X1tFq8M$SF;UD7Z*VS<YIO*^rInsfIQk0)O;E2wt&_6uiHv+AAD>meb;z^wQMqT@q#C~l^Y-'
        'zDK_(Wy@NpgUFgwNhf%IoeR49V|>e<#w+1QO<yn6rq?c=BKU%Y-'
        'bc>m+`w=W;>58gd}y#L&504o=hkXXq`{GoLMNw%zfb9N>oi`-'
        '7qLX={d_2^g3({h?$$b43K$wdgAVg42IX;$RF2zBCD@oBYM*_qi+PUn>5_$oO&&PV~oOfVfdv5&_9kYa<kaSbj`P'
        '6ehEEbL|>kyNwOiBzQKGMUR+x->C8Mn-'
        '$nv*F}?uBl=4r!I(idKZwXUez4~23N7RP7d1M?H!XrixTqOy3@IRt8@FiPQQ2SZrjVI<?};b3QL`z`Gz|k_9Ak(O'
        '8=6Q7P2(`wfzhTr=y};KFMYQ0R_zlXGvM|%k_hPZ-*YX|BjME-RmEkK!399@Qc0rgzi2s?INZXYpI;W0bw_ieNZA'
        'M`M1o3Bxg!Yw@N!)H`#95L-)pRJRXn6J7i@xInPw$S*z}LZ#%;DPA2iJF5afpv-'
        '!L#^Q@niIJ|Bvv!8B<?hse@&MW3&?S}cISR-'
        'u1bhk<hIN!5!HnKg=wKikzwi_B;$TSTNVThP17f6HcmW5n(@ClYNshBDoS9tT43+&1AX-aN!o`xj4kktY<_K?OTa'
        '|!6dNQKE)ljODUqMYYv=unIf;UZKGJHWs8k>O2T8+^t;^w+C^_l~$7um%=XuQVV@Gy40`e_cZZeH0xiCn)z;p`+c'
        'S#N7s(yADL4F#gwUI_*!O{P?PblrBc%(?dO!&tX?F#IGGj=-'
        '=EHMnWN<1~&~q3v*D*tJCnBXm?QKsdE#6rG{wG({8V?G)3DJUY%FV-'
        'g!P*$aK%F>DJ+z)X%PPTdhAD(ItVU>nl4Wv2xjw*N3i1(@R4xBfaR?u5}YG(4(D5t?TvAz}T48Y@k=+;1VY2*pA?'
        'YrF!VmekA_nt-=wiG6<Oj%_U3TEF{vUn<wBa9yH;2O?yv5G<-'
        '44oX*^XDURvVyL(tx)D)L)Hoo#I85AjMflo;<1M$4ZDU5v-eqim-'
        '`piRU90Hk7@#us1+Ce9vPJFUQA3|dhH2kbjK-_Nf1VnR6G<*WW-*Rb@)1E8Vp6BOfZTN~-'
        ';<7?*v5&kxR@p)HUaK7mWWizMhn1@5Pdw-#rIOImL%U{F|E|Gy@3u}KUZnxA-_UB!7GGZxlIod(I=#W{bjvTk(`h'
        't6;<1C;jGLS3`%jXhM%VxB8>7FOIhj60vUgcyjh&Tel&AXSr^yn+yHdzQvQ`gw8eeU)?5le++UrNG^QspL{8wY-'
        'a>sXK)Y@LZR9G(W-H3T(R`0Z7A)%gFG3mmOcechmcLVQ=-'
        'MPKBwKKNf6|2^NjjsEfUG_B!?yXS9<xirL)V(uey)sh!vWQG`UX%M&bj=h*%?E7{kbFX<2AcaUi5YP_;A72L#d#R'
        'SJO)zn0LZGsRB}LcTYucx7un&eZxya@$VYGC&7VcMOV1+Q6VD>t*`{X^zPnTT_JP`Frz-!da~>2<u}A-'
        '3eGftAmEPnvgkxILK4&&~B|&GF-!z?kKa@_sK8gI@emh=;q|Vw)8|2F#%j~>M*<(?N-'
        '4;yOC#Ku(v$z~;Qv%1h@0Y2EA#AR6HNT+MB$V}k6?g=*4))e{VgDM0-'
        'O%2%7WxoYW03&Zbcw)m^63s@ss8v?ydq%V5hdh<D>)a1MKDB43OJk*SN72?T~@x3!*>ow%7zM=P|CQILhJ-'
        '@G`)=X96FsT^?WiNcCt+C=7o{%kf#Zsy*iHu92!n;_@R&B0~KE8|Fuu}(EO}>!KfBX(9)Z_PWW_<0t{}w(b8Yly;'
        'St6sIZy;ha<O$yG1My`PS-Vh2{=NXf0&}HJXUMgy60pjhWI5B@a8Udw>-'
        'Kt)GL!Bm(lmr~4ZMq{kW$*qwQh7>CBe)}~+EBa?a7?0PSg(R*ZDB6ITAN4zjbk3*!itEHg-'
        '#Wh#fo?NTQoWv9`7{B!%g;wI$8;x;h4lqmi63%om6L4_Nh8ek9Mlnpn9Mo;`GI0&rY1$kk@Nehgb>Wp9SpoGJEFc'
        'U)4PIQw3>e%5(ZnxITgjU^@F=iAw{%0X5b7sz@9FjM(R1#d%^V~e-'
        '(P5I3*<tu9K+82@VjfZrx`!lw;d#mFH7L?@o9;*mD$4+>eUw}ia-'
        'B*F<%89j5!ct{f9BF#^2<NGX7vb!RqGDqVA(Iy?BE+WcYy=SV^P~+vMhLhRuGWQ^#5r%X5L*t#C#4lBK$G9}_8=P'
        'hCR7L<S4|HD+rTVU2wNgXPG*QN6TkxyZtPnvYdJD5>rq!)~g*MaS7?I(p?Kp1NNP;y8yTnM@4~Q3k~OITc}^ukzS'
        '26r1K~N<=NueBrOPQHWNg!%z8*+FI^-p+W@tlBUQJMQ)qMpU#kin-'
        ')4epmYrQOG#d8D}}eT!|8UpWm<l;;iKp$UCDSy>6k>3?+-%@`N2nwSlFhq?2XVq)px0ks-'
        '!0W^WW@XMi$B_5T;)Q=$+IP-'
        'H=HcEXhFNBA{d;x>L34tmNARnTXl$9<_iWQS_#X`cnxa6t$f;I=BuYzREGjI&lFypeqM@5bLwWJXR+-'
        'y@bU?2|B9)S$SEGAIUU|h2m=K0RI+4?u6GORS?|Et|P4YR>q_Rx-NE4zKK;NlnP>xiDwFWlzHt*MthpjMN&+GY~`'
        'bX<|pCG8W<u~A>FwoQ`KW0LuPfQX(vA*`E(h7bwqbJ<DE$|i3mm5N@UYE-'
        '!mjFENH5WcIC0j5&X7FRpISO>^B%?$;mV?(G(x>dnBz@TwK+-nfdoK?fW%;7hcDX-vxGxPX@Faz)dD-'
        '+(u{ogIH+S9WR$)JjB41<7^^QWyAn_fS1^Xcm+g6ODK)h4yTw*k{K?%!-_OTB83-'
        'XpvEx7>x`+q&ykr%<hW$CD0Kh&Vk#bW614;4qQsB%K1mJp(pA@RF0w@O_1a~`wYZsQnlxSL6-'
        'tw8MqpiaQ<_Fm*74(oL_tsh^4eY^$I6Wy_LSfjZtE?slE7U#okWn~rDcc^TwH69m?4<W7XJ~ZU_cr2^;NK!6+1Fk'
        'RPkRH=pDH(tZTIkg7{jzEGo6n4;cOeYX%cwy1GO*{6Lc0(E<t0t|)b#LglliRgGkE9MC<)Xls03ZNU*(V6i%KuaQ'
        '@Jfr}xSlw}4DPV~N&?$WEnIYd3$>Yg_Oci4v^7W|5z=kdiDfphs}iFsN`PQrWzWH%_E$znqKmGV=k4=j6t@;i_Zz'
        '$<c2ls_yRm_C(Dipl3(`k0)Zd4|>E3TN6D@s^rOg`!Eoppo9?owL4t7j6Fmqgo_R1QHqyvX*LWT=6vtuDmIBDWSX'
        'WiPv3~my+jCFa?dav{_1cAXj)FPvr9~TAnBafR-twBg<7M*HP@J6gbW=Ih1UQnj%0^K?JAi@B`E_X<~$r4TA%vJ*'
        'r0Z)#F9pb6v8qYE3kI*u22tGjakLO-r)WR1n%qIBLh6+xY;9eo|CppB+P#az-icuh4E<;bLNCcrswC);CHloba|I'
        'S*#bSS~{lDO<1AFilz^eIi8eZ9y>AYxNN^DW9GNxL+V?3A>c^iwTeU%CadWRlMlf+7mbx{eN;QpKZxhfgvb|*b1z'
        'n_JwE(HyO&QX&p>6?jsMl<na{ne%_$1n=o(^fI3$BLVEQRAoMM)jPl{qKVk+pu5Z_}8Uz}A~X3$smg>5vG8ltfIm'
        'h2HcH||Hd;$696cIv<(a(CgE=v$t(NPIJ;=!~J=3*hsuXOX_WGQJkVAdl(2bP$J#k=~WrfY9!M0onM#4#s77)FB>'
        'o8AzGr(ozgAB<X1MwDg1%#6u%l<+g;+EsNa+Yphh3!UOz8QTb#vXC?*FmwI|wbl|1AX|udGkpOLOBG;<E<t9&nux'
        '9eDpwda<v*Dv1OcAch!5(h#QK{Jd5z7lJPeGUIX&kpmScFv#u*|KGlTuY*Tn%W1-'
        'F2^C4{@^`g)6q%h8V8Fc14{@D3Eub(S+qWGO3Hgz>yKGfdlh=j}O{MEoQ7GQK#PSUj1#+hyM|tWbTV?jowuWqY_F'
        'SFRd@{*t+tJ4p6RNUSa$$*<C)t@~Gz~dM8E`gL4B{Jp8cQlO-'
        '8>Ns>#_O!*0%5fIQ3WUQ@7>)xgYyV3A$Q6QPnsF052-HC@JJq#Z{F1Y6n3Br+2b{8n6oK9U?r8v;tUd-'
        'b0wwRyzEn`$W2)$(*a9&W440KLP%EinN+1!TJ(2v&7E6p=Sd8iIE=QL|6sMXcg<XCzudcIWVn0_Mecv?=bX&SbjQ'
        'O**#Zk}nXUthghifPetdt*?IvrAv+^2g<IZH8%VJQ8oGR1Io<N?kvjE^ohChQEgQn9dERyHeJA6GYd))?N1&6yK5'
        '8<nFca%B04nugpx6*UH<niuAQ6>8OP1PnQE++^yN*tSf$<Icr#?puvzTU<aHrg^^FiS}Sv;F>#xE?zwpWrOWDFrD'
        '-$mJLI-^qawD?B;dsxJTKp{e-'
        'p~{{K{0CT{^+Dr!MKHu=dfY;5D(sGggY(x_e`GcnT!NM2TC9g*HwL%&qp(ddXaZtkt${?V}=XrrBOYopYt*L@6DW'
        ';!bXVY&+LOwl^mI`G)H=8&n0_jz_O43Apq6s)1BajS?qjk*lWG6X@=FSpLP7+f590Y99rWF?uib#A<c2UXc&CVNB'
        'LC-2_7PShs6m?~+ZJ7}$8omJ%8sSjQv9C)CSoGIx54!Um~3KgrlspdTe3^-'
        'p)#d8jL@QTH^tzLH(Go8}o6Y_(P{RGceo9d%yj=Va-Sh8FJra6HL}ACzCLPB;0AHewT|h`+tiQITPLPk<0c68p-'
        'DD)eZB`0aSU9=*z=udhrTOp#1K&=!k-bkDMrLfp7VwCAa8k-'
        '|V2ucTv433bZxlF~z}SEI=CY)25vq5(D<2>Y~~d7IycR=L`m90@+!lE{|>X}#MuUuY$iev-RNAWIL(G6)0{^DJ2a'
        '`z8CEr|k<y**0rv@s7Ii8dAa@E+T$Cz;C@})Qz?d#ow3oH*_R#<>{k*;aWTz%DBk{b5_IQIxLll#EkT(pA?66XZ;'
        'X;cRX17br8dAK5VayJ$y{;YP7G~K;dw<H=RSr1Ob~r*&91N3*4HtIxk=U^!(-XSMRHG>il|!@gIHuA-'
        'RNRJaawg01{eq_ABR=M~_w}?KxT-<%^-f1n|g<@4EH-kICmFlTu7s3V9RgfbR5~8Qx)lMmTJbmTcf6JxfmVDLo>j'
        '9|yRULbnqw-N&9Ir>}76upJ75m#<6l=Q1{-'
        '8_6+@%{mletjHxsVo{vPlK#TLk;?;(B`z%yKeBW_)<l!I>Djh0VpYYHz(4E|vU2?AuY8v*{cs0fu3vT{`0w(t?kN'
        'r>=NW~ejfUjchocy;FW5R%|1-'
        '~pL|jaV{(&<hQ4}+wethZ~2*)LC7WhoLFEIv91flNST^IT6?{Tp3E%jy<{r!LcFR)|VO3o_6`jJ)xj>IK%qn0ci`'
        'I%I>zF=r|nxyG?`#&<YT1ET%Pi<{5DMF%STJ5a-'
        '5FKX~Ghk?aWCbbO>08Cx3Fvw<Y=~Y5iYNI=Y{%z{LYciG(46udR78WC_A;%4-'
        '~O_yCEsoQX2w@v1zobH9~XXU8H$$Inn!mvR83^{s<CHs`#-'
        '}>E`QrIS)c4cgn{lx2loDR$ZI%>vjPv&{O1lF1HK%o)A9Q}pUgAsbll`veBW^t`%OlfQHq#PA82-@Xy-'
        '+lVY)8r6f|vmM0a)6JnZaQR_Zq-nEwOm$Y=<|U~1e&3l>eypKX67A0bE8PH6jNo^_OMw4ckXQT<9Kuc0<^Rfkq@q'
        'x)}R{@}<2>>p$v8?lk9>z(U|{k_VScZllTiuLn*6pK61Wo;{c!LwO2X5;_pxMd7i28KS%i&PDHp39YKy$d5I7T^1'
        'FNa1<`xJmxRgRy&x;r1BISq66NEjDY)`4iys&`Qk6iBig^X8(DChtLV;@9S^DN!GH~cVA5D`g-'
        'XeF%~i;sTq#)T>U9<ym!yVu?odfJsMoyuOpJ^G;4ZS&K?1b_;lie1zDa6BL}bClU646{;Em5D*n<hs*7~qnhrJ$k'
        ';wG09X;HWtntk^Yid5&i`>_wg=mm?Dxa$d(RTZqxcdz4>>loG^OWFC^-'
        'nM0sZoaqVYgj2X)8r(7=l!6*z3r4`7{@gp}L1bkzFiJ99_rCed7la?UQ(GRY|>$<j!{ZQQ?qmR!xF03Lm|3<<1fg'
        'F8<m?#S^n}i7TWCR{Y=p^}oWJ66=^WSu7r7w66HHvXV%uSBk3)S<GC>6DFybh5w>7bNw_khy4;2$}rOE4eg-'
        'Dd-)PAc;SZX742Zd<@ZH?_D`VR`g4AojnaLof7&{(Z*74gC)0XC)4CE42J+f_19`X8ZwDUlMtYPg&Amk6up-qm<y'
        'aQ?`(2swUEEYzcc{|A35pv7X0QOV>Ba^eblM1S>J?raFCE3g^&SYSJH~&&E|%ULz~oil9B<`?r|hI+6+IPyNB8dj'
        'o|et`uw>XX!7*hu5vvRjwH5CX?~8d9g!Z057PG88Y<Ad~WC>;03j+R8PGkjSInN=`s;C;aSjCtTCu6Q0^|ozBnd3'
        '!?`G3o<KBxcb+4EXA4iS-AqtrQ!L&2CL{3q^278^P9AZmgbWcA+AFE18zYK2<vw1a@Q`Wz7bx*nh%3((H>0o}6z-'
        'TMeY_bfp7t_$cmnU2Va%fLNh-_`)V69j5vn(>N^L4H0S_agai4dku%ia)7aC(IcF#AUmlwxf=JwbGq8&}7-'
        'IR62XQ4sU1=y{5HYk;el2aUnFSHx#wvi)$8Ehf6P9uH#m#D(KLzPQ(t%^0n>oi>~T$O;zsK`F>da2}P!Dhu@HU98'
        '{R+)7WHv5FfC2@z(}EBg#A=8=@2})452)X8CGf7dPw0Xk*8j=O-'
        'tVR9kU6k*K$)FlpDgfjBp{%Ixr%1!=Ik(UBT^(RLs%0EVUjbhX{en_d<uz`kQL&h{FZ#`x6`XfK(!y0EG%8ABMw!'
        'eLohwx>I%kSy-v?2*d1&S(Qwg_XtwQT4EeWdsy{L?J?lqZVwFq0^d^G~Ap#Yz!NSb_W(lhX-'
        '~ao>J+!>u}?iuO~YGLKe2_AS6A9IH2aHQk7Aiq>n<XCx5R|TQw@{RM!3cq{~b}!@LlWwC27lCVTEFZD=sbPng@qf'
        'VIU84|R%Eg_8M_bigP%E<qKWr;&E)Q_jk1ftszxMCgnIRevDMlaq8HnldB;*D5z7_s;Anj2CDl&5;5{=JR9p{Bo9'
        ';i$Y?9Cgs2Xpa0iZ|M%5saW@c&%K-'
        'jwt3Mmg8B%K;KkAVIu9K*2<E=s25F2FH5hAlaz*d}d%ng}!<y@hJTltIlDyY=J58}6q+>uhlZJE)rYO_rFs=6an?'
        'D{{EH@ssew{F+o+5?NM2bfhrw*!^ZPnlPkhtgGDtyABpuoLy$E)1QHDoW-2Va_ZyPXb1%D-'
        'A3ct!TS_XsMvJ)iGkyvb7$GWMq`l^g0fK(FIIUl%I*mrH|7wnz+&Eg#Pxz$N$KVfq#yV2%rsahJGNzBVXO^lqSRH'
        'wVV@gc+cUiocBG_S(51u;|=_cDY9q@2A%IU16FUL)thI{#%V|TX=p5@!qo0L+4@)-nE5Fg)VcmL&ZZ-'
        'M>Y#tW0%$EgM&fvlJ^9Sro$vZx`<I4cXl<`4AV)~bbQ9h{Woh7AgKI;9AoMDIX00>2&jBqyf_A6_OXay%4;(_Ui;'
        '!CR$&VY(6`^zLKoqP)$2E}qMCXmb#J4xn$Uiyh(6;N|yjwOWc0Id}uGhr=6VTQ^vb{6WIUIjn)A|fkjkq+&Q{a4&'
        'Pf8v+@AeR8t5D0YYMWLf;6&~>>)*=};B}6nQI)4Qe=u(TfDpK^YT%)0_1b*%O%yowIjcm8SKJuM_oB6yT8u4bD6)'
        '!_f-B#Ef?NjI9XAAo9U(*-495jRrIVk4W-'
        '$HrS`HAO$a{g(b$l7O1<_Swe+5y!xub%pPnq^UyLSR`(Xhxs{SpDsBIn7b?g<U<)h!FDxcR!2halOfH2aWcl42JR'
        'd*U_6HI<fuXPWqxu5*nWdc0bjHb6@_#b?8m)+Kw@-'
        '$j?~97F?`&m4;P&0^5($3*u+_#lZVbu<V$DD!*C{8*32car{gto@j8N4H{k0Q|Z!)PMTVi$rGbgnkeD9pT<?{`6l'
        '&kiC5u?bP<>-ecR#V9f=#wLEKYu7yb1GeQMVZ~OEo&Xc%uGu%oixu=V>#f09fh)(FFA&NG+sHPT!%g`-'
        '#eUqhsWsR>>&=oX3OrgFk)t5#$$(m%uM?ko2=7sPFDTPNpSm1bRNA9jfANmDfv5-FcSJo(<-'
        'ioZV)yr5>q7YeVfe+YYd@^WtPnhG|+L=Y>kXRSC?}*6`!-'
        '*%OY$<by?$WiNVSxz>ADqD1(zcTw;a>9te1>x<YojWzj#f^ZG3(0F63-ub&xl>Jia&^-Yp$hk2QR#w5i_W-'
        'M2Sd$u^u+JH;6lbH4U&_<>glV#+P!~>uy+XO42$g;ya6nYj2t)TliN@yuiL&;w7K;Qpt4m1x|OPz+Do44akh<7=6'
        'wwr1dK0vwQ)oXg?dzz06`s^BX5rh6zx20+8yGgSk~dDlJ&(YKoxJOJa#Mf_~gp7r_I>a?;QmL-'
        'g*sHrK};+ZZ7du#2f8G~9*r%p5(z{g}+k7@-V;R>UtynoQBnhz&7HrXzH(13Z$VOffKFftTY^`pHa-cy!8u5q%qD'
        '^msT2^@(s0+ZkX8)FOP6$q<zc^xrAb1m2SG;{MNni=+|nCi@J>He)#^g{6d!F)s|zA;;{ji!<o(ZHitq(-'
        'Uc@EY$!(Z=WoCI>i~RSMugp$;t9DpYJxay#+ICdr1X_FOS@)H^`%ACe5$~<M;5DA)j5-'
        '%wp?sLsa#a+Ub@S?f}~!=q?Wr7?{I#_oR-'
        '6(!&#`rz9MeR%5rPbn6%~3wqWTzVPf9C&WKkfxRayK4L0%_R(DEmCo&HAX$rip;hhD(7qk^R+bgzNU$SH_9>rup|'
        'oR?S4|!=Y(u}PF&dQn27k@jRJco_@%KZoJ(|P-'
        'G;2M)VRD~qhHt*(qRRAn$y3o*x^7$QmOS!EqSK4<?7`>?JcyWVUIaPYemG$#5*6f0$Z_Ub@7lQ`Hk={kdo4C%>{P'
        'pA9`A?MCp21T4G(BSj5qNtCV@_hA6^PAg%T~6qqtdROET;%Y_1#GZj3jXQOi2Mpww)~p(9z%%B(uA6=^vG+9DGl5'
        'oGMgxdwI@vHsKP(H<5~p@@g8L5toya^5IpFUI)}$Lh8apJi>rNx(vK=UIiotU1d{r_Ad)JX^P70zLD|&#;%FM*jS'
        'g-qYciO0r~74$)Xe)Jw6}U^<z2CswK?<+=s(k!%&c+TW+c?-'
        '?9EVGlab5>#DWIF!TYoU#>2dB`wQg##~nW2lmdu&{4n^$D{QDtRypUaYA~u&Jtj;~GCxNfNzDfuq)Op%NJ~n7}!p'
        'RSj6Vzu}A)L)p+DuW&v&PsKb9bY&;mm%QO_&=9Y=twBjTP^eIHjQPZ@`DGwAe+-'
        'h*c|r~#_8q61cT?WBQ<gg;envKej~KV*Z6cC>$_65Zt&CilgSuzM?f1yd+OB)#CYY~>d<uFg>e!-'
        '(udnnHF{yGZY%e7I*SRnmpA6$~a%2hr^U{<4#>K>_j+nB5Ykw8aKuK0mV8}2lEjKA^c15A-'
        'CM*h0)5)ld$sUwU&>w8!5XR&k$s;}AEE2$Qu<GAoqzWg0*eu|5P^97-'
        '&Y>oW0JAumvfIO5W}q8(qFZg103y(iYsR6Ld~*6|@4%~qfBzq#+U`lshmvko5ao&GcO(NZ*w*AlT^xOk_Xmm-'
        'I`JKaJ(8;4$BUu^n?<epKGMECjHj^1K-'
        'FNxN3mi1_8i&FzA<b*K|b{OnXD(aI?(CI>i8W3?F+g9NV%r&?fl*Y8nyT_?eG7`e=%phcHGh1q(=cpTd}c3&;I?t'
        'v%!ZMc{C{hd4aJoXmYHl>N=<vo7cl-xR%)oLd6ROP-UG73*#)B3|N!($T#iA8brg6*k~q}Kv_EyR~W1tMHBF~(&j'
        'dIW3#WsxxltBb;U!A<%}Nx*zWKYT>%nkov(V~X=R~qRR@N@5$M`Qjcbrb+68FM)^$L`A+QW;J9IVc`e;gnr-'
        'I83<Ko##29VE*lY~{EZ*8biR;m_cI9rr(QcsiNsW74ap~SQp8NSHa9Scvw87U}O7p+}sfm?VlvUf#<I-'
        'F#n_Kr;c7PchN1V)`0i=eRB5wl)o^HaHrmAg@&L~{u{9CVE#_~*PM#A8d=00Ay$`wUZ6^kOH|tQ`F$`v3>N(>xzV'
        'Y7;|rvnkRSpISmsP29uI@WEditP;<RSikWR2TpnTW}#<<kwIM9l!tG@uh1ox6N~k>Q5aS~c;DrwJ7i%0EPr1MuAs'
        'buwl0csO((Z%IuTyGrmOY&0Hfa@1$tCREn%w*^`|Y-'
        '12<bEBim!!Tz%b(NNtVNaMg!vMF;wN&466Xr0Wyw$I+Mmpa0e<cCgvLBza%DTF4MXUouyr=F0qrxK&?5lQh?N&=R'
        '{2GY!@INU!&eUSsI*G%@Pi1l6sXA2h4}=^ddBx@3N{d*2Q8UGE9aM(?+ET%o<O*1JN7U7)?om4?pA!YP?G^lt0yt'
        'Cx+yT20#(clN@Z;9uFZ@Ai#7EBJgD_Uw*qT^{t%-'
        'L+uGH*a6OdOvvj<KwsQ2R}Z4@xzbrq2ryte5uN+^ZMnR$8TS}d;Q8(ynSaoRPoh&%uoB57th}R2$i<KyBGa2x_2k'
        'K8Ex<MZr^vV&IA6OnkJ)sxWFAgaP{A#XB)aRXwe-n3fLks62*7|Ov%CEA-%fUD^m1IZMj#2$z(!3%G=|b-'
        'bBcx>GCi(PsS=Jx@)XJ`xyMooDd`u;=PG3d={W$7ps{>$-otUM-Et-|Lj0q^&iS?>Xho)eXt9ZINSzFtcA$dRaM-'
        '#yH1ds65(|%O(OP=t}6@{^C5Z+$1%%~CVf3y@q777N1*82et);$r}96)f9fKD9pR<Bt4Yj?9QKN{0p>1Tp^@k7E1'
        'q&h*Fyzu01Eu5-J9q8Ic89n<7ltMqvX)`2R+KKuk?V7jd5m8E=>c^7UaE=kSbF%O_XwNA-'
        'cblWk{%(;y#|-Ts>p=xJXW9Xsw!?p7id(sh_5=pmhtcG5jZ-(qu&Yi-O@kIm%a#sO-lDjq`lo7>KCMfe5f}8@`By'
        '4}X{L;0`?}3wsi&(k8$~jjo)D!@%RJL;IRwJxD~x2ZUeQD;-'
        '3KIw+#X28gW(Ni!fcEg#+y3J~CGkxlB`XiGlgDp*>+-'
        '^z^X#IY!l@xoOQrR|i;=W_*Ofn7;}77jZ+8za8v>x(|;nNDgP_OP<H2?`o8+eB@=OHI5$-'
        'S+ZJ*_}~(Ncx9uEcnxR>c_7kp<EU6eky!z_9D9>yJr-'
        'a*WL0oolIi;PNrf*wr*GajoqN2(|wA(o`ifSx{}#B0=sxe;=A%D+;@a4frP1e)XAME9`a$gR~yq~aU-'
        '$*DU%CeX>u~irgOco%a49r(J@m186NA}iZD^?2m@VRHeA;{8N)hJt$pgWx&B1eD3~1>Lqa4o27QwximfJ~=Zz);9'
        'n~;Br^RnL6#0`r3Sep34y`5=mg7yJoGS4TLc?tr1c|}d8@M=QD7Ii(^=sNNwbdP#LcvZsZ^P1Hsu~OQLFIXmm$2k'
        'Wgon`A=V^hPu=(=h3^yk{dz)`>#ce6&yJR)3*JMRMp**YiouH%p0gqA9x>8(mrVR@&I~i_IXT8PttG0mi;3fvh%;'
        'pboJh*wBkCyoF>HKW+@J1ZRH*VY*3_$H13{WGE1&RaAELU6<8H{v*RpY}O|1bZo+r0'
    ),
    '_portable_underwriter_dde01304924f.reporting._core': (
        'c-qx{Yj@j5vfy|A3JmuPVN<l^I5T5iGbb63GdJhPN!E7mz31rYU?38d5R(7{fU*>g=5Ouy{bgH^epEjIQc5PfJ7-'
        'QT0^QZs)z$Ut>cV(Dep@e#^-'
        ')#qx+Y&I`Q>F(T;^SoG{v@Vy0W@VR!zN0@^w?>%iE)NQ8&dh*}{LyMOW5UI~$M3qY*XAvemBJHAR*s<p#Sac~#Y2'
        'PVGh`^=y-'
        'OSKNM4uh)fwl3y%zr?>fb3j<G+_r=e<qFNN(dYO0mVx6~b(dw4wk!ewM<)+ZDfBf)zY94ZvEeyUcFLa9^3@+U*j;'
        'Wu2RozaLZ_BO#q8P-h-DZ26<ZV)I^`mWGE#U$Df4kH~z!&+&y2y6bvS>av2qjxCi{q2ej$gcd{$e#Vw4FI^uDZ=y'
        'BhVE1B6$64huGBD#ipn_J<Md3!2hzYEV4FV71^hvyu9j)Wp+_k?UY{Hr8d2@ZGOA1^W}7uJ_4cQx?JL_x;VW3>c6'
        'wMUw!jk_U7xa-@N{i{q5a1f6adQ{+sVUyq}r|?|*#v^;fUoWMBRFH}9R7ZxqaT$T4lF9^T}O?%TR;^;-'
        'ofFRM4J6)>Ln9Cw2I=p7;bx~{t7Mgdhk3$%dNSz(%|qtOpd4J6onxZM`XTyxLac-1f)-'
        ';SrrxGk;=pbq_gv1_VqQMcXreDu0r?>5!<&41==m;iKdyC(UE*#T!Opm%o;Ppa(<SgvXETlleE&fWvKqJ{pOZQhh'
        '^UHw><UDzKOdZ70x`{CX9fBE*!+w8AzzW(aRZ$AKW|Le`WQ@s_>0NONBILwnfqJ)^(X>u|<X8PC}o+i)DLpxP$`u'
        'TKpZ-;#U=IvMCefZ|}dv7@B2mI%}XJ~#zkDH?1f-'
        'ouI>G<_JFE@!GzHMe%8XNw=zd))j`nz1?`VibVH`}@eAt&E1cFlE>-'
        'O#YR&29}6@483|)iP<X@>u&@uRRTaw5WHmKV_Hukg|hwwmf#|lw&>Yj2SB7wSUrmlQ>}BZG762cY5ai7if}DlLMx'
        'F5o`Q=u5UQc`_X7bdMx?T%1;(PZ<?lVChvfppu^}#ddhSekKg5G3lsWuRa8lybj_~qK0%kOTcy2|1?k`9q5vsgFL'
        '#S#$=Z7KrO})d5a7QQ)%-'
        ')VE7CqBzTc9PHUs~)Y`1HWA)x(?$p8(K#Z_G{Kzyz0CaEiJMDUo{wJW=IaZ0jjj!RN5Q0I~;g|^pa@oB7d4oT7sl'
        'rl7T<*MKZhWV+0l`TL8eJYw$7E&PGzQgh7^ba?yx7o+jq^wXlo;;^#%X)`8YhA87Qh4U|F^hUyMfU~2S(Y$`@?u9'
        '?Q8sw48W>F(0oxSB53t<;XLu8IhF_pLSmnAc{;Uu)!alpG>ke?qw;9lST`s`b3E)MWw?H+6*q@%iWVFkw-'
        '0U{lqFAr9w)~~Apm+h7#R{xEVEZnErd^gz+G4#*lcO()(x+w%&`P5kg4Z+pF+-'
        'AVC#h*tt`cNU<_YtZD1>~SkSb*MA2YZOFgP-ERaZyFW()GlfktCV)*<+1mGo%0J!3kM^w35*OpVZ-Ccy2MDJ@FGk'
        '(ve|uIlwQSXK_d4x>MxOHD_+jbZ*_jtY0*1WZ@858Gr1Eq<uW7FYq6^195?44XI@%}>&OR7JFm#&EHEz8@>y(?RN'
        'Q`DT;%qV)VAl<X1^PPq0!q~w*t?$sg485piuG0TrUyy(`q@HaN7k`o&1xxf!)UzMX1QYq8cs0ETiLQl{`UkLf{tr'
        'QA+`clYm;7B3=Wgv$92B8=dEmD_~*GNNP^~(ahy{;jtFeNgiPCF%FP@>TOM>+x9D9R+o8rI{)>7k@V*}@b$WR3}q'
        'HATUc;?4`-fRJ0$E5_=T9@RxT$^fhJ9pZFv7Cf&=9A5$t)T;zE%eu(n&7A^yKTZQ|aZbpOA%Tfk7^yFY9G-'
        '&^4ZvHR!N*1xz8e@~2=U0m2bf1Zv4gM<VaL&qz?UdnWvAuq9-Sz~hr261?y_B&#hkV-yUC`+bADUV0=fK{s3@MjL'
        'W3gh+fxDc0{V{kOxlq4m!hc=p&o7qEXEFSixZj&gCxcVqyEi`ucLdFUzn{*<Gf1t3g})OEE|z|P?1S`UFi9`3S^K'
        '2Xs&@Jw3i$Dx<xr7-'
        'WYY%!Q3zoyYCTNsHF(zm!LbhMbq6Hn=WphSmTbYYKsv19z}<KRCh&w5HMWXDMbs@!>39=ACZ2BG4kD-Jpo^lyPCF'
        'aJxh|eI}H}M<IVze;C$qDB=jfYXz9aUp?5_K6EU99k8QEq$+)SBZt>8-'
        'nEAS{E?am?Wcl|GZ@*1eCD_BpE2?+hc85W=?&_2ylB6mNv^Bt(-!5l=1r)z-'
        'Kt8j7Kyfo<Ixk`(TQUur`cs>MfyJgiJwjQa;(QssE-'
        'x#L>2iyG^vHk%OAM(lzo`Ie<r1!N#0aJ!=y&=0RD_+pkYel2z&2$38P5Q~s>m_U486PEc3p7a*=%<1`#U>yQWoG}'
        'wgC*UReLCfXvJ2mws6o=9w*R`T%-'
        'ks^Rl|hu8T#dWp{*3Y*CSBJx$yoAit`p3r=uEJJAlwq=#J;T0qcTh!1GOs$023gvN|fY-'
        '3X9KZ;w=9OA~fIXB=vz;c}?cLehfja19p45Sgx=-0Vsad-'
        '^4=Plu`5nbYTozwPA^>m;uOtIXMmxIz|T~rfznBoKco1lyN?YH7!WsU%1QMJHxc+3HMc$0UFE83C-'
        'Z*0a#DgmZtyFwSYprL2^dObOO$VFXsIZ(Q;z*S%{Oq*5>qvR5R)*N%K_7(WmxUcpw%#*3JzhaEjzscOz=iI!_H?W'
        'g3z2RaI?XELnWem$soh{)7vUyufV>=GpIzemMwpJ0R-'
        'dL2{4sm%D+95<qS3;8A!Fz#%2?MJ;re>arL7nsokb$ZeSkWp&ra&8&<kSSnHBoO;aun4M+LemB4j;N;$b!3_5Q8G'
        'xDx%`JoKBB)a87DKmt~E$fblasO(g&-'
        'XQ$w1ghijYW7>WCNK|ls8lB}%8?$OyC4Vy01&vc5h2)4IN<n9^_96I)8R<7xQB*Tj4l8Bk-'
        '#B@i;3>}RCvamXEi^2a6CZY(racTPwx>DjAdbX#a_^3YB2=c?&>{j6mK<c&2H-'
        '7FE!0tJkFZdIneiWr?kpIF`Z*E=kGkY<AzncV023>CVDg(;yKlaLH$#(mz&oKqWBY$9yDOI#+qy036-'
        'na&sawKlXjG<2bT|dg3^7maOq9@9MmEbz6`dxP#}Ab4%VCgSI&)(nF#lmEp=4%Hs4@A1!gCxOQ`0@u$et&!=KJWa'
        '<91AfHjIgxnWH>%BYK+g3~G*c!oLFh4-fn9M>?X`<ezA7Pn8};Q74lSKtF6+si8TVfyhmC-'
        '?T3TO*YovXs{kkoma3(-V>50F})^VDn*p<##siYM3#;1{(`@tcQL--gTg9~19A<JwP>#KY|9?~fGB9{HBgrf6W31'
        'D(H_7kx+41!F;Y2+q7)rdv>Xs3yt>oGnQYGtn2Bw>es11RPd$VyM?=<jthhdoK0{#0!1fulzj7qt4R{ih$v-'
        't%;LP1-Vs(XQDTPik`8C%%R{$x|<ke*{vF5S1t<jRLx<9^1Ep-SA${5E*5W2}5-'
        'uEYZklgQGL_G{2Ow85r+PJLHM>EeBIv_k5wD5y$3>f_Tu3VKMQq}n?NW+@#Lf2vdBOwYBoQ24Yf~M{ieKfR}<vka'
        '%4-k9g*}9%ru(-'
        '@8^~F!H5uFWY0EF(CdDZKb?G4}2Bp&Xdh1_p&@i<(~0!;q0t@4TJl#<|h9J10lRSc?C5{MBaCopcHFz%XLZ{t!xV'
        'v2u*8)eoA5MAXc3(H>$i7jkD;Hml7z-8(*rMOuX+b+QaaY7aI&*0gR_V(wZP}cOn%G)c<-'
        'x{Z!?g!AMkuNb#!A8=SMtEL4I&5=|V~K`%Tv?KxtxYy}Pr$b69r&Kk<jZQ&iH$<kZvQR0gf+Vh<{@^R<COa@rNx~'
        '1aLxj5T`w%Zp~+R?k@&K>nJk-'
        'n%P}*1T*j(?E)1?|L@~DWJ64~(Izpdm4^NsuG0mgxpnD6RjnFy&6B}RqDsx9%+V|0L+FVE!$<tElY|~yK?L1fx+q'
        'N(MHdp`JmTUpfwHfUj*p5Bzz4R%O>OhQ1f!q{%#c?VpcsPv&CH)w=6Lv;Qf$5xIs_<N^A+%~yQZshlEDM-mMrXd0'
        'o)D^v*I*)U_&CJP4H&hH0)#m#RXmqaK@<+6$|(vvoM%LsX;b481=55tm?j^A>*wRPWAP!qT|1avFa?JGJv%);Iyt'
        'xCVd4Z6&363vjGnp;j2o)>InqSAXmdq6+2o?$;j~I^H8t58Zq%te{@KatIjS}HA>ztVE}MF{WdMua1}!%W7efyxw'
        'w1&fnqPo`!pyh{y|jGb&MtNOo9P_fKY8NOf&t!`IUmoCrviHy(=>VJ;m*eCxk{tz^0mduMUoq87I2!PL4N8V%iIt'
        'as~yPjb@{QF@b4<GQWK>ZL|DxZ5SC+|kK7?{_3pX^rIEFmjt-'
        'h)(GKq~D^w5LWMU)96tt<6kft^hB85(nyWMr%#g42OcxtzjI*jQFRo{kzY9<4tVv^y&WJ3Fq#1rPo98qN83Btky)'
        '#8`BTuwj?+$fi1TQrz2odM$d&w&Z(e}|A)VD@8S#<N3ftUwuTKvn-'
        'j4wcpcc)I`yvRKz`p*v3#Vxat@&ANI$KPiq*K2Mca07^X5kR{{5i>Vkv^H6~~<rx5P$?TX!Siu|to`xCdgNg!GqT'
        'Yir6AfmD17#O^3+jyWO6pG;C*$%I<R&$mM4cXHB$g*Ct&jrbfqmmB2$@AWRmP^Cap4~%1k+Qx6fxA1pZ)_rwP$2V'
        '{#0y9(!a=Lu0rNn%ruXwpHgrF=_G_Nq~NJg);N&}C4h+d4(xO)!K5w-J3BAm)Wg=XU)g-'
        'pK{*Y=pOJ^+l$RuRle3&@Pdz)<b?qH|WkzgMfL>d|nyx_}p4k{2UKPOL#o75t=Cet`gyyG0VK4}1c?aN0Mw(p~U>'
        'G^+VaV~ec{B}x-'
        'y>nOYQ2_o(f>m=vM#UcdYP^AO}W0^(@3Xi4z&Tt0^~!>>@w7_t#^$};&3&h)8OqN`2WC~xT!5wW)##@8GkM6y+9t'
        'hj3FqRj`~9jF{+7C`Pn`aiVpD<*CDcs&i46UI*VTX%=94Q3a0nAr4&#ctcyw#GQ&W#N<~0c1VZ$!32z(A9oz$YtF'
        'I*Vh`Iwns69Lb5eq8TWudz;UyKidPI*5^UQ4P$Gv{}ty(3x>to$8mzio@%vd)_4&^8`y3$~Yx_d0@1`*t9m9F_0l'
        '5L5h7_?&r4Mu^Kh&)jjtAYEjh$Qm$3Vx0)pTwX%DxlbZ}3}XnD1Rce%p-'
        '%D&A9r+FFVeTZ;1dO)TV@)&Bck234~#47puhVd(j$;X<I_++4fi9l<94G*!@aEBal6{*!~IzTxqbPW+wVReb&_V#'
        'L~ZLZCh_wwuex$wDCZ>-'
        'ofwP)V3J&ZQLf9bbkaEZW}h<1$Q<A&y}rK=WJH{Q<r6@A6~%g(7r!Z&lo#yy*Pee2oAh}0xo~s&xj25s6Y;r)^~`'
        'H2992dBGZU+!4CU>_k_tT!4b(&mQqH;%t@uPz(Dz)gp4fgeWxsuZSuSoFsa3M}nrFFb$AZ!n`<uDPtB*DCz#&{>&'
        '>o3&kyp!4<+8g1vxBbJme&Qv?;Oo`neW;bL}~VMS-3htsJlaKT_y|Xkk2b|VcE5Bm=NYf&Q*$IywZ0w1Kj}YGD-d'
        'W6+AO&qW<b@Xl4#gEfw50E=f=lIuE?oW}lCF2}5o9OSz4!(8+jve1eC{+mmDZ=X3t&rTXXPIHI!1J?WD&vH$fg)`'
        '1J>APhSC1T{7N=X3neOZv~t9z=T{^I<K;7^Zv9(gcKi$LS|#38Zewvkw)#W*dw9d<O(yRrE+-y|x!W8QnZ-'
        'H+790?qb)LRnfLb8t3G13g_fw(Nx7+vv5*?wF(APG2;{I4_Ad{Sv0Eh#DI8TuCEKQOn@%Cb&>4otP9WMTCi5uoLr'
        '74#(?WmWs-xDMB;-kI@A|<y7bNee3!I~d|k}&nm*8tTfk5lbpgu<M&c3^gkvG-'
        '2yZT%atQ|9bzNcr2{YUO{&$ioH+bNKmr#$Gq<{Z=0!Y;hhK@3A4a!O-Ojqr$A!`ngnpUuh>vqO-UzK>=NbxH~Nm*'
        'ietG27@_z07}@|h;Qh=Cd%#+t*8!jMVn3qjfM$&=@xm5kD9Vaqi-'
        'n&e~y#9agTAR$#e*=TnsIC!u_sPiK>bJNIC1aBo?)y%HwoXza^XVCte{AQv@BNJP7`}sNS-k<;Uxr472CN9!b-'
        '1yIFyZ`BP1%O#5BX$8G6NGhyVA0n-'
        'nSJ)+r8@h7_efDrF{ah#zwpjoi2T$6oZ5uwFOuVP3y~~_YQoT|l^Teg1DrHC5)+~u*d8$L=hFmSia#>$b{EZ2ctH'
        'yW`h<FV(jFt70OUzBIRYXm!%3Uml*(aNk*PLAAZYmLpMYo@4vejgXp5*KZ;WTu;m;UzRgwv2B!r_g=PCL}ICGfuO'
        'LJ&OzPSnkU~_V~T+9hoK>aha*$h^U{SG|WCOX+PdFppDDV);dqtT}fNSG!8QnZtfk{1+X_OOtcP`jsoyOF(`4HAR'
        'Q4aF}nj$i(9O2d490vz!|P*Y2S<)uDxT0_5WzHqUgdT^h@BJk$VTUe2dyIz6mQD1_8U^Wsqo)hq3ADiw(fwrCHEu'
        'PFyIMv>qm<YoefMuGEcrgg_*~vK&jsHA8SK^_(+~k@Bmi?Yt80wT^S?*f31fblJSnfoR5`2t{xhTr@<OQv^g|4$_'
        '`NY=EbUI?_Q$5O41QlE-P|A-w7B^emZJ+ZAgA#`t^fPBz+4tF8{Uw}+6^8^^qq3t?SJ!yFJmDqd9#hJRn~$-'
        '%@{U+xqZsG4k&w>RR8H-j9CE4KEk)IF-ZRfdEfXG3LI<^Gr)s7CNAe#<u}zAq-d$d)i$;7BzB77KTM-O7Umvk-'
        '_@%}?kn5tN`VtCSo~(*b;KTz48=3sq!YY2`3r;wrgh6a}i>n0qp=f6E1hQujAeRU+bo~5<S_p^G$~^<eLlI)?(Nu'
        '8ZG+7s`Zce+up)3g0ReZ((T3sqid*EdJnH!srN3ETM1dS7L2fp-<91RSHZ-tG1;;IAV=W|Z;LFfMuJ)i7zCi}Ajt'
        'ivcG0?G#EFwTt*FhBY<)BKf28H85ugbsf_$%rFbO*eFk+;dc_5LXALPPE=-Gk>b-'
        'D2UzM`F><FMz7KOi}&VU7+vO9<+FYx{~z7?KBu3`K7rTEkze#C@9uxS*Z(h!i7R(q<9pucxUcJ`sD6R@#P&?Y2~{'
        'YJ9^%LD@n3mOg3yCW`EpLZr=tN6c6^y*frkGtCPQh6duG!Y{(ea4``$@BV2;ZU;IaMZ!<xk|ifVDS$(xU3m%k3<t'
        '%0T3|0*8K>Ro|8RuY6ih=u;|;OudXgETXNv&y4AQJMH5`XRa!Z}0}MoP3Aq4O>-'
        'c0|qNWYc~bE*h;~H32UII@<c?N4(fQ+E7-$RQg$g`8)4NI8A_HDG8c=~*k7O%Dom1fhc-'
        '<FNF*~_?E;%(I#<~@W(MLNSecZe^OOfk*@Rx#mnYkadUB**`2+lOXoP?IcM>GQo<|ABAYzmP6bw=^LZ)_%fx19g)'
        'U7HiJ=9U+9%2s=*=F40wKH7Ura?JXOqM0)$IQo^zdA;#MAv=I6E79L+HHy^@9M@lE}N5yK`fv>5?{Qfhc*b%<cp7'
        'V*~~(HmPly!?<4Z?MXF1Y>vNxrRxk7`^#5N(F5+NoZz33vL+^{ao?(M?=e<IX-#8YX{-'
        '5u!5ybEa_o*`r4z6G?E@MHRrpafxLH{Ur%wTW%=(9M$7YOkA3lC!2$F!!q*#J|sZU|mcBa*`txsK+PrHWUhQsNVa'
        '1=kAG<tHL(n{jhpW{%|r;cqa<i!8rv3MwY`Xi<Ae)?{C#8=P}2rD4Lt-'
        '<DXyvBFbQ3VGtR@fL+BKYO&wsk$I#@0!9Rid}K^tN@l;`;H2T@gRHhM7bdkiT=7tY_e7IrIz1r6eb--lR~_%=6$r'
        'c8dXavqoV{i-'
        'YB}Mg?r@TAOQz4W`mdAq7sL6&m0ed=iwR$xv2X=IT%U(GXxM^_<B=TN(%eae;S*^Q!|nZHM(@9c<=LQ2vZCeO<4&'
        'zlG<qHuQrE>W-'
        'Qj_cEUnO5awCpjB}*fI(5VaLz{ZQ^wVS8M~MkA{VFs!L7{;FL+4{+O3#QC11AfVPM{qJm&+;hMZIWw2|UPH>lAM%'
        'K{aB+#ZJAQJZvRkC{iTy^R;k11LV(C{>+$f-'
        '@p^PbOW<Gj|3VKlFZb1A*q|b2SFN6QKfLv)FX%J*q`$@+LMNAXK+9cswLdi35WD29k}EVvk#CYk_qDXh6BoqIE4u'
        'o96;m)as&@a-IX-km$K`V%&W6T`zsSV4a06y?>ZR9RI(f>=EwC#2Zj+F&YyTd#^89Fu|Qu&v;U%*f4-'
        'c2L~`Gw68rp!Mz}gKhsct7F)C+~C^JkxI4B2U+mzS&;+CuPqnG`-'
        '{DZwEv4#W0@w&i0{<5{&bSjuBlua>Qa*x!BMDL|dW3VaH)f~9-'
        'Q4(IChjIA=;Uqe{l<rM{a(@=0?;p^M(>xW+aK<&~0Dcef>I2mbXyFCMmc+HXZ>%MdDIur^?D_CQa8j<~Ebj2CF3V'
        'N9E)t9jiKEZ_0@$9#r6Q5-z-LH6h{)Y=hkwuZ5co{ffQCC~daKL}c!1)U5jl^^J4_)_x(z((s;-'
        'b|6M@*PzGV{ueJVhNkkV+qKt#}X!OOX-'
        '@&ffS$C<hS0ymiZMJPMWBT<)ou~@b^E0Dbq7oIQO0UU(E{~kH8{96KebQO0qT*9|!2HN?#^gs+T=pyX{AU*Aa)He'
        '&hCT`JvHFpOk8EjxnBchFBK=y%pAjh<fluh*(a#?(;mLA%Jh13@6##wNAt7tROL>cq7y&{E^h}_z8Dw8!VVf!1AP'
        '{eaI=ZcrzTQ~^k8g;JLqogS!Sl?UhD$B5{o2ZK1!JDwJdzFlP*=LdV=sah<pKS;G(7ie>OIiQQGp^tLE*f7C?kT4'
        '4zc$754^WT4U81h8B}Zq9;W=H}AD(bR_<nYow1MxO`bnoEK@IviP{6fR$}Q?XyGM+oXn|M`eXuAZ{)7*&GGC1vMC'
        'doV85P4=UPY_it7{?Ft~d6WJByaJEU)gVie<YPdhP84ayrW6etD7|xf=>kzjTA!IC{N3c77DKAzX`D=P(VNIKfr}'
        'MSIy@%`K`iw?&gNySUZma$MJi*{K)lhaujJyxApT`{9{{Im-'
        '{)P3}b#W{}^V5`EkAx1_~^`!q%m)_cD<QQ|4}Fc9eaR^CEZ;-'
        'M#a;d#mOCS>1j@&0>n&WMNHaQmItE0m77US^5t0Cf@*Y%cMfSQ9DJ*bs5;<M70z29$G+%<;0Q<e6UsCRn|hsk-'
        'S|Q(aD6^TSnI1_P1`lpq;H&vTPhssMerLybODh2z`F@l<GoX@YDeR@<*lwyU%+X*P_4K8<oq8bic4TNKzYrKX89m'
        'RZrT%3H`h9Lzd1WI6A@f-'
        'NTZWuQ}3Q0$q9{+_!h66Kf<54vmUR%(R=3MCpNmkyA_4Eb1kIBF#zhD9kt5cA=IZC+x|Af?-'
        'ZUK1&O`m{O|70^ZYr9Q`qy?!!vMZHdIq_H1Rr)9uoio^-'
        'xrB+O>Cd!qLNEt1?SrmudYMrv>bzXHjr;;<sGoKKctF2zS1Uz^1;-'
        '@Ct9d@j~p)s)|(B+I{zMb*XT+awAmT;U~{o)M#kqzrR7(})xu#T8OEN}7Nk#hcQdv-nD6?cZ7&qJ0Og?tTEsCGqf'
        'Dn{sX^%6NeL5y?`7SGF_xXkE0P36SamlK)6Av&jiIG;j`J(1(+5D!ile3<RCC%G_ipXcFcR0fO-'
        'y9T_EVwFamWlH;&sULv9RB5E32Z}Nz{7{t1!*v>t9)s+1Qn-'
        '%$d|hwUbku7>(j2Hlv`5rfM9Q%tgE#Idd6lnk$tHDgx(C4S;zd}2bMSI)lz&9AgYoGhw@%f0Y-T3ucLTR=PpYj-'
        '6jl3Y5YoUdQ`d}@Sz*L)Ke#9aOox*NO=3#Ht$5-}t)Ba|PoCWM-r1G4le0St&fZ6+F6tlmLc*^oVShgny--'
        '*dkx<u+U0^>xZLI3P3H9G+?w_itV^Lv@C%Q<J!R}Ep<NCyyyy$K@_x&LJwD;|O|K)7C<$GH_R20a5p<q3vCkd0g`'
        '$I2&-w((fRVEPAca-#*(W8so3CR;xSXWhmR5v5l&%EU$2UT`g-'
        '2F0!5aqWF*UoZXG(09o&Q^HMx;mq~{bN<`!%qVZGE~^i;?_1jC0u<(Em{(2KesDEi-%--'
        '9DH(~(gnNzZmOsOl{ME#Iw25!vv~lJA8k}X2l?Ng0S(zDXZ?7o!^_@5h7dN2CzeOi(>mktKg&K|6ny}MN+vq>*df'
        'qq=c&F6w+~{IS0C}KC9KOXi<+wk>S95Zf`PXu2N%J`CZ-xagdRn0Q#V>xRiT0QN?`{<SAmfeG2`;ho~?Kv+hFp4&'
        'kh9~*pP#uhlmid7^Jc=tTBkUUa3Lz<9zgRopUfF%iOKgJsMb^{zNIQY>-'
        '2F%riYqkZ!=}7|^o)lN_=+_S+r3w#UPl+h6u!i{^YUtaNV+y)gnGX1MG9;M>E@mVJiAzy`bD6TUOlQk4RR&2G2Ff'
        'zbjP<=!(y#95RGhAiP8u=e9Df`bD{?_ZHad_z0JwkbDxb8E_g@5SFk&U0kBZpXn(&>t6gFkT^DBbddOd;lRQtmR_'
        '#YTl3TX(a6Psc86R#vkpOdjYec%M}RIppQSvp@Ezs9Q+2yN(Lr2^7-'
        'trneN3pI@dk)ac6oM%8|JqLLzhXj1dI=wA0zqhiBC~<6>?0*>O-'
        'wQ?_|PuC3D##EBpc>Sp<Nh}qSFK&Hag=C_Vp_Z}R?4-'
        'i6O@AE;}g7)j&?%L__=5$e(1xh{SqqU70rA4U<;*P&jeK189Lx;<;$M4~YbRBdMDt@KZA<KQT%CGBPGYp>5esu_V'
        'pQ`->w0jAFnbqrEi=WnWBizohK}6QhW3_KD_<nR)Y|nE$dri@?vnQdMwV}o_@op^Zac}t0lTMf-'
        '?mz448wx4t96s#n-6unkK0U8`Ao&*n`t4c&KyOel_Z!sV(O*oBP9G8%jJQBp=~n-'
        '}kA|jqx!E@LwNCc$H^&F#UjtXe9#PiVNM&mkKHG11>@ktYuCX&X*F9ccj3;`nyXolA0BaOlASb5(vao5~>fJ}yh)'
        'ZTFChy}$R~2y3Oe)Yly-32#UR;x@Z@aLRroyQFXHU!6$@%y+dZ0c%=;M2>J!J6TOnVQGyz7ZrB%|-'
        'rV$@@fY7Are+dFG>FLpEycyb`9<K~dg-Jkv4C`Wp}?O)?q%)GB4wZ7}+IaPB{<M!WHQi1FV)q_6uzP7DPOZTKfa~'
        'bCBCtOfGvsPiA)C-;y=t$_?R9m(H=#c0XN1?&tU-NZv@v%$wC3^h{=HNch9ezUgV{x0FO4yjJ#;Cz7Fi+tpVn@Kt'
        'N7(vDLqa6~;5?LY$g}7`&KT$Yu*GM|0?0mfOogg+5FVw6m5T%5A(Dg8I5RE)q{??1<)^B2(c{#Ur%||A=KCL*1_b'
        '}bY#;inw1Vo}+j2}(r$v1xl1GewK*_l!4NSpJHQ#~uuNy+iz58Il2IO2mc#CpF+*R=I3dpU~dsDz^WYcBTw^#8PN'
        '`F}$OFwV&OQ7j)Nk8}iVk+xWFIUv6=&tG|HQ|Ex_A4+})J@gjdTtat)t@U0%7x*oI?ZR6R&i%2NX4ODkKwRBI5U<'
        'Di+XztCR4E$pPEuAxHHvXt7S>$-'
        'Ex9QXKwRz|En%aR9cc?3&sJz40$@$t8hNULq_!}Pw#@AyCiz4&ie`}opB{4y6opQa_aDV<(sI)j5^Kq>PPiU@=ac'
        'X?L;KV+5*39H&NFy&2wbvvU)>*YeshO^dELD#@4T6xOB_Yj51UI#8%eabL2K>V|p+?*Tqipv&BZj5f5+!Vpkr^l`'
        'JioJwS&RZ&(-'
        'C4Exx;!F5!L)s8SyXo%Bo%TTJHHz!peQnIIR7~Kz%@414_bXkK;#*fkGW<IU{h$?B!4H!zrQiB#B&_#ibNre33EV'
        'Q!DLF7-'
        'u`<JM9!8(AINSzt_G~=&Z<0z?rkI<~KA_eXiY}oe=)d(fkUx#;n5TDp@8|bT6_S4$jKTXr14(Qw(KMK9n`vR+^x4'
        'sZ<QDC(gB4MshV-ZK(<rsIk$eR`=2G0wl2K0w2JITY{`-C&B>s{ejd4dR6-'
        '^R`Ye(*nLpz()JobRi@*9xmW_Aj<mFe$?RA+O(vHM#-'
        'f1mv*fcBHYs_HGN6>O@{Q7ei3f^mijs55T}Icw}mk#gDW)GoP4v5FfZ}p_xL1nucGvhz>1h=k>8%<S(V(?7AYng)'
        '81TrBO!?CO4w^pU=G`KeWQ%-q5Q-8Z?UYcu^X+*X(fmeeOwrdKi=bE-'
        '>v4Yhe7_1I7>c05i!UJ~`zU(tnD#!KX*WJGI~r@rw<na4%-'
        ')Ph;WSafb3Ck)Qe)>qb3&MN3@BqUwydER6yAPIA5+$uXTyBifu_N!;|ulN`*vuP2QIRfAPH{AXT!Bl%QY^E(2Ur|'
        'TMNK@>;nYP}1T3>$?<&tZ@ms{!sL(g0w_4<jMzrpP%QEnD-'
        '$*tgZ(HIsehtKM%>>AIT{`>$`wV+LTaZKteTPim;AwkTG}mvoD*;!rE<B`TLU4&8KDgmq=NMt)KQcmgiJDO<k4j*'
        '?S|8?RkcsM?Nfze8iJnH9=k3+p>VG+C7$T3!}kuM>f@y}st={fK~JSw!Qk<9E6be$*O#x}<V2P-'
        'fx3Sp@<NyDZv8Q*Nmpx?06emxR{?$C$*qYXK~lbqC3U))#AgatGB2^~Fy(!fZVJg`J_*1zV#lsX!a1EEaT<aOnDu'
        '@6;0CC;$6@B??j^APp}4yP|~!YGo<={XF2x_y`mHuonw&{SQE+p98|t_&Yv)blrrYr~Gwxy@2CU-'
        '`NH*;$Fu2HS3-)7Q4-'
        'Eol`|CEUBqlJ$Cg4JXLyB!5id(H_<lXvLEl+`d=M5y0`Z6;ro}nZx~!3{3HTZTmr>}^>nx{X8ZrjdYi8=WhRaDu}'
        'fWSwk3PN254e|4zIWAs`tiku;t%81RtLROyAxf{p|4Js-'
        ')0EUJcWPzp)iaa+U=#;jh_+#RxCwujY=>E9hHae^>0=wEH%!yIzMG&oe^KaW(dBUcFMhE~+x8-'
        'Jk|Z`o0)$(E~_2WSSjbW#_;+^uyzK(es>(+e|Iz!QY^KY<QV$5oPAdg|!tWO<(ag@>vTPZ{UkgUB90g@OSsA7xV3'
        'nlsbomw#{#L{n42=pw0UfFO}OV1XI+!PH`Z{_RKnlTso5<D@X*@T(`QH5lbq*rs1W@7k<sNAZR%XZ;>Hkj?{8#4?'
        ')X;KQ#BrRgQ>III1i02*6GtUo==nf<F{;8WCYn6WqqLpKAQ38%VrjDbS654qXc;!?>1`XX@W`Np|tM6!oM<x`?+H'
        'fZ4rt!k%-'
        '|y5>u>jm1lQZW;g5i!Vo!AN2t~LzuZ)ZD?hKc|ZXdE+A!Fs#iVH0@*SaEm+rGrh*a<Q2jjS=Q}WcKVMscHpC{Ihw'
        '%pdNMx4Z6yY@jJf=g%k)FyjhpKRujpxv3u%MZ&ts{pt1B69NqCuM|-'
        'a0gBas}Pz^86qgOVVNd9`HKs?1L8v3xnWwsSy)d3An^<As2w2WXm$YtZMoyOJJ)+Q4YGrb1M5<_ueNA({CiQD}aX'
        'xp52YL9>+&B`Y}~1UO%KC_q^+K)+rJvf^{wfetA=YZjI7Uzw=`)QY2_eaFDj{6%Se{9-w)5Reqh9Yfjkk@-'
        'wjfwuqqr#%I8G)?WPv#@^GjVK|MGybczqVS=2kN+g;^tOki`AOPGjnxN(kY`5S7ruq;UTA+fEHq?jERgCXGxIwBD'
        'Ks@xsn8EfzaEpAaKT(cOn4HZxeiSS|7sh`8ieeOaEPiP=zSx!PC1;H;*=)$z(_jj$8d@%yvfvn-'
        'y0H6eIyV2l`A<<cO<cnA_JiA6ZA9lpDh^TW>v*ks`b;A5P<M5zGD?2P^;c96+L-'
        'F}NgXq}9Wy+kE!s|wtvkNsSM(_qB_;R+mHsSZ19*pa*hN415lfs!y3|`m)F174Gv3jnea~<FP9$-'
        'H#_u>YMOT0RFMNfcl`Ph)BlKdx$ODJRx}wShC6AODMtPY`d0YIMzEp{yt5t{s9{MlX4}D0h-N~jIn&{WcAIkWV-'
        'IA-uF_Bd6ou6|j*l#ZWo{Hmi!Ekfumu;k3Zm$0jU*)AWp7THUvA3Xe@3-'
        'jB^Pc<UlC}ADoOhlutzPqQj~2RmVVp2MH8CW8!fTUL?<lmdx}%|wXHxglgKu@u4=Y9EpJ>nbW|4+&nNQx?fS<N)Y'
        'LJbajH<x}9p&3;ShJFoI)h&3f&LDGm-eaHNX<-'
        'gQ|j&}qq66irL;!R)zvAJk<7#bHW3B2bYIwjclY6QBi`i5vLue9%dHkX_0^i~<9PbR21qc)KwpQ3h%8Cy8I<=y3E'
        '@ad+MzGf+3QTb)nxiU%kbdkkCm}K^xK|J30}PWPjXZJnRT{8ww;XN^~bTRWQw``VVg+G&*cxvTq2op9!<jxm?D+m'
        '_2ZMX`9NGPHs=85Zy`8R`;BpBpwWyX?m^F#)CZpDWUoww9g|}HhP+zt+M=1HdTAymPnVm5uee&3H#5qU0Jm7>NZy'
        'TSSKVe!(G>V&d_VMIGBGHBo~)=SZ<|1%&>#@1N}QUG`}s}ax*W819ys$RRvgfiUQGSGCM#oOedPN?LcffGWaSvte'
        'N1nPrDU)a7%>qQUrFiuBVvqLCs$EF&gWd(ix7P+eAGoz22(O1Rv6s>nk$zEg(>Yij0T}0)s>rKR@I*-'
        'KYn<f&Om#v@H4L61fRFtVlf}L;O<wLmmgrU%|1q#5MOwWgKZM)xF=X@0$J$I(4B$ohj-uq<=Z!Jv-fY_e)ZjlZ(h'
        'GY8)ObyA4`UNjE~e;kA3R!bi*jvtTBEhTUpr^@V-=(?lp_qRUrj}L(eoUJu3AlXH}zOuzA!&0WK*q$If*K`No-'
        '%bT-'
        '`!6ylBSfjNk#>J@eoMLwK&F$Ej_I&DB23C1f)=&YH&%O`NqPeT<U^nBF(74KuEQ4TMu*f%eOT39CU*BP@;t{r|>m'
        '=?zUmbqGhUgWqneTGJ^w*6uRg6#c(#G{s301GMdNTcOI!{q?u#hf@rF2hC0XW3%^?K!tn^nt*z>A!$!?uB8Z&KYK'
        '#kIS-'
        '|s6z);aILt3U76J%Me4TdL>_)d&T4}RE=?xB@|mGoF%fb_u??s(usmS%;KhmYu3H`b+5IYNPY}bhn=oYN{I@XiSa'
        '#3VzxIU{q;vbHW6!eUpAV@0K7BqvNZ;dqq%6a$gZQ=dGmgmwlO<zK9W0Og;wZiEnN<39Z{QCaoR9tw<K#Z4'
    ),
}
# fmt: on


def _new_package(name: str) -> None:
    if name in _sys.modules:
        return
    package = _types.ModuleType(name)
    package.__file__ = f"<embedded-package:{name}>"
    package.__package__ = name
    package.__path__ = []
    _sys.modules[name] = package


def _load_embedded_runtime() -> None:
    _new_package(_RUNTIME_PREFIX)
    _new_package(f"{_RUNTIME_PREFIX}.reporting")
    for name, payload in _EMBEDDED_SOURCES.items():
        if name in _sys.modules:
            continue
        module = _types.ModuleType(name)
        module.__file__ = f"<embedded:{name}>"
        module.__package__ = name.rpartition(".")[0]
        _sys.modules[name] = module
        try:
            source = _zlib.decompress(_base64.b85decode(payload)).decode("utf-8")
            exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
        except Exception:
            _sys.modules.pop(name, None)
            raise


_load_embedded_runtime()
_core = _sys.modules[f"{_RUNTIME_PREFIX}.reporting._core"]
_evidence = _sys.modules[f"{_RUNTIME_PREFIX}.reporting.evidence"]

UnderwriterReportError = _core.UnderwriterReportError
UnderwriterReportOptions = _core.UnderwriterReportOptions
UnderwriterReportResult = _core.UnderwriterReportResult
build_scored_model_report = _core.build_scored_model_report

CapabilityUnavailable = _evidence.CapabilityUnavailable
EvidenceFact = _evidence.EvidenceFact
EvidenceRequest = _evidence.EvidenceRequest
ExactLossEvidence = _evidence.ExactLossEvidence
FeatureImportanceEvidence = _evidence.FeatureImportanceEvidence
InteractionEvidence = _evidence.InteractionEvidence
MainEffectEvidence = _evidence.MainEffectEvidence
ModelEvidence = _evidence.ModelEvidence
SuppressionMetadata = _evidence.SuppressionMetadata

ProblemType = Literal["frequency", "severity", "burn_cost"]
ColumnOrValues = str | Sequence[float] | np.ndarray | pd.Series
ComparisonUnit = str | Sequence[Any] | np.ndarray | pd.Series
_ALLOWED_SECTIONS = {"report", "data", "columns", "predictions"}
_ALLOWED_KEYS = {
    "report": {
        "output_path",
        "title",
        "model_type",
        "tweedie_power",
        "top_k",
        "double_lift_bins",
        "curve_bins",
        "distribution_bins",
        "movement_bins",
        "comparison_bootstrap_replicates",
        "comparison_bootstrap_seed",
        "minimum_cell_size",
    },
    "data": {"path"},
    "columns": {"actual", "sample_weight", "features", "comparison_unit", "offset"},
}


def build_report(
    frame: pd.DataFrame,
    *,
    actual: ColumnOrValues,
    predictions: Mapping[str, ColumnOrValues],
    sample_weight: ColumnOrValues,
    features: Sequence[str],
    model_type: ProblemType,
    output_path: str | Path,
    offset: ColumnOrValues | None = None,
    comparison_unit: ComparisonUnit | None = None,
    evidence: Mapping[str, ModelEvidence] | None = None,
    title: str = "Pricing model review",
    tweedie_power: float | None = None,
    minimum_cell_size: int = 20,
) -> UnderwriterReportResult:
    """Build a self-contained aggregate report from already-scored predictions."""
    options = UnderwriterReportOptions(
        title=title,
        problem_type=model_type,
        tweedie_power=tweedie_power,
        minimum_cell_size=minimum_cell_size,
    )
    return build_scored_model_report(
        frame,
        actual=actual,
        predictions=predictions,
        sample_weight=sample_weight,
        features=features,
        output_path=output_path,
        evidence=evidence,
        offset=offset,
        comparison_unit=comparison_unit,
        options=options,
    )


@dataclass(frozen=True)
class PortableReportConfig:
    data_path: Path
    output_path: Path
    actual: str
    sample_weight: str
    features: tuple[str, ...]
    predictions: dict[str, str]
    options: UnderwriterReportOptions
    comparison_unit: str | None = None
    offset: str | None = None


def _required_string(table: Mapping[str, Any], key: str, label: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _config_path(
    raw_value: Any,
    *,
    label: str,
    relative_to: Path,
    suffixes: tuple[str, ...],
    suffix_message: str,
) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(raw_value.strip()).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    resolved = path.resolve()
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"{label} {suffix_message}")
    return resolved


def _features(raw_value: Any) -> tuple[str, ...]:
    if (
        not isinstance(raw_value, list)
        or not raw_value
        or not all(isinstance(value, str) and value.strip() for value in raw_value)
    ):
        raise ValueError("[columns].features must be a non-empty string array")
    resolved = tuple(value.strip() for value in raw_value)
    if len(set(resolved)) != len(resolved):
        raise ValueError("[columns].features must not contain duplicates")
    return resolved


def _predictions(raw_value: Any) -> dict[str, str]:
    if not isinstance(raw_value, dict) or not raw_value:
        raise ValueError("[predictions] must be a non-empty table")
    resolved: dict[str, str] = {}
    for raw_name, raw_column in raw_value.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_column, str) or not raw_column.strip():
            raise ValueError("[predictions] must map non-empty names to non-empty column names")
        if name in resolved:
            raise ValueError(f"[predictions] contains duplicate normalized model name: {name!r}")
        resolved[name] = raw_column.strip()
    return resolved


def _optional_number(raw_value: Any, label: str) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise TypeError(f"{label} must be numeric, not boolean")
    if not isinstance(raw_value, int | float):
        raise TypeError(f"{label} must be numeric")
    return float(raw_value)


def load_config(path: str | Path) -> PortableReportConfig:
    """Load a prediction-only portable report TOML file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    unknown_sections = set(payload) - _ALLOWED_SECTIONS
    if unknown_sections:
        raise ValueError("unknown TOML sections: " + ", ".join(sorted(unknown_sections)))
    for section, allowed_keys in _ALLOWED_KEYS.items():
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict):
            raise TypeError(f"TOML section [{section}] must be a table")
        unknown_keys = set(section_payload) - allowed_keys
        if unknown_keys:
            raise ValueError(f"unknown [{section}] keys: " + ", ".join(sorted(unknown_keys)))
    report = payload["report"]
    data = payload["data"]
    columns = payload["columns"]
    title = report.get("title", "Pricing model review")
    if not isinstance(title, str):
        raise TypeError("[report].title must be a string")
    title = title.strip()
    if not title:
        raise ValueError("[report].title must be non-empty")
    model_type = report.get("model_type")
    if model_type not in {"frequency", "severity", "burn_cost"}:
        raise ValueError("[report].model_type must be frequency, severity, or burn_cost")
    features = _features(columns.get("features"))
    predictions = _predictions(payload.get("predictions"))
    comparison_unit = columns.get("comparison_unit")
    if comparison_unit is not None:
        if not isinstance(comparison_unit, str) or not comparison_unit.strip():
            raise ValueError("[columns].comparison_unit must be a non-empty string")
        comparison_unit = comparison_unit.strip()
        if comparison_unit in features:
            raise ValueError("comparison_unit must not also appear in features")
    offset = columns.get("offset")
    if offset is not None:
        if not isinstance(offset, str) or not offset.strip():
            raise ValueError("[columns].offset must be a non-empty string")
        offset = offset.strip()
    options = UnderwriterReportOptions(
        title=title,
        problem_type=model_type,
        tweedie_power=_optional_number(report.get("tweedie_power"), "[report].tweedie_power"),
        top_k=report.get("top_k", 12),
        double_lift_bins=report.get("double_lift_bins", 10),
        curve_bins=report.get("curve_bins", 100),
        distribution_bins=report.get("distribution_bins", 200),
        movement_bins=report.get("movement_bins", 10),
        comparison_bootstrap_replicates=report.get(
            "comparison_bootstrap_replicates",
            200,
        ),
        comparison_bootstrap_seed=report.get("comparison_bootstrap_seed", 1729),
        minimum_cell_size=report.get("minimum_cell_size", 20),
    )
    return PortableReportConfig(
        data_path=_config_path(
            data.get("path"),
            label="[data].path",
            relative_to=config_path.parent,
            suffixes=(".csv", ".feather", ".parquet"),
            suffix_message="must end in .csv, .feather, or .parquet",
        ),
        output_path=_config_path(
            report.get("output_path"),
            label="[report].output_path",
            relative_to=config_path.parent,
            suffixes=(".html", ".htm"),
            suffix_message="must end in .html or .htm",
        ),
        actual=_required_string(columns, "actual", "[columns].actual"),
        sample_weight=_required_string(
            columns,
            "sample_weight",
            "[columns].sample_weight",
        ),
        features=features,
        predictions=predictions,
        options=options,
        comparison_unit=comparison_unit,
        offset=offset,
    )


def _read_configured_frame(config: PortableReportConfig) -> pd.DataFrame:
    required_columns = list(
        dict.fromkeys(
            [
                config.actual,
                config.sample_weight,
                *([config.comparison_unit] if config.comparison_unit else []),
                *([config.offset] if config.offset else []),
                *config.features,
                *config.predictions.values(),
            ]
        )
    )
    if not config.data_path.is_file():
        raise FileNotFoundError(f"configured input file does not exist: {config.data_path}")
    suffix = config.data_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(config.data_path, columns=required_columns)
    if suffix == ".feather":
        return pd.read_feather(config.data_path, columns=required_columns)
    return pd.read_csv(config.data_path, usecols=required_columns).loc[:, required_columns]


def build_report_from_config(config: PortableReportConfig) -> UnderwriterReportResult:
    """Read configured scored columns and build the portable report."""
    return build_scored_model_report(
        _read_configured_frame(config),
        actual=config.actual,
        predictions=config.predictions,
        sample_weight=config.sample_weight,
        features=config.features,
        output_path=config.output_path,
        offset=config.offset,
        comparison_unit=config.comparison_unit,
        options=config.options,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained model review from scored predictions."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to report TOML")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_report_from_config(load_config(args.config))
    print(f"Report: {result.output_path}")
    print(result.metrics.to_string(index=False))


__all__ = [
    "SOURCE_SHA256",
    "CapabilityUnavailable",
    "EvidenceFact",
    "EvidenceRequest",
    "ExactLossEvidence",
    "FeatureImportanceEvidence",
    "InteractionEvidence",
    "MainEffectEvidence",
    "ModelEvidence",
    "PortableReportConfig",
    "SuppressionMetadata",
    "UnderwriterReportError",
    "UnderwriterReportOptions",
    "UnderwriterReportResult",
    "build_report",
    "build_report_from_config",
    "build_scored_model_report",
    "load_config",
]


if __name__ == "__main__":
    main()
