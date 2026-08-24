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

SOURCE_SHA256 = "d85916da56a6174f1766a093c7f1e977ecbb0a6329cc95b11af4dd34a6675ef5"
_RUNTIME_PREFIX = "_portable_underwriter_d85916da56a6"
# fmt: off
_EMBEDDED_SOURCES = {
    '_portable_underwriter_d85916da56a6.reporting._underwriter_styles': (
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
    '_portable_underwriter_d85916da56a6.reporting.evidence': (
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
        'M%UMQHpE(@+Y9DB>29#E`hEvZt+tHX7MxmU&IyuD<H$a*G;pBf8N64AMNK=aV(^e`nfJX71bU$6tcsODI&vuGDAy'
        'Zw8H9=qZmW&CpqI6rbkT^iJsW{1Q_s8_P5Q}5~^!X?WUa8B{BHklzoA9WoMnREtiRBZn9B%=A~wRV>+@r)TSIo?G'
        '2FnTu_>mL9GiiUbQ1xChi_Z6wBEP(2deb@@?ITxu-Q1TY*>+++_hu-'
        'M2J6=++XuSG{&AbIV1HdNq^Fzi4Za4gNZ}RXR(;WoXKr?#C!g_DP(ywUBU{9w0ejvN&GwpY;@?A7THM{{kDZWfIt'
        'xJ#JoZ+G?8@U&;YFis}Nl(CN0^6{mflOVxF^LJ^!jYIc=DGFngfB+R0r$rBawZB^XVAS0_yHwyY|vCgQ-'
        ')mu=`1sl+MZFPN8y30{1s4Mbo+^;SZe-G<wotw5f9m!Hf_e|~xc6Yva#i82Y+28%=5>zQyCtBKznW(yB*~-'
        'P?6;7h%s3=086ep64v{e(!xEv46HB~4MfBOP(3ZGMr<Nyq;`za8uPFwmc^BYUP6VcJCR#1WS*$m8@ayX{i@s%z0D'
        '%Os?)EK^9x!+kc_#f?`!Z(>4DQOEdH%O19{k~BKB)2DJd6qo;KEzo<zZO+j0!4e^Auz>2442#Uwr$$UX4D5X);eW'
        'LIe1I6RT5+wB-5EcDCn6GE0-'
        'X(A2mH28|}7gg|z1!PP<;um1!XmN7F+&P_LXZK}A73<j@ctIp*1H_DbTK6!lifqy}gz@vzS7&sBe`Y4Hp!-'
        '|cBb{ln-UMrb)$pA`@uH<@Bosrpwfx?NH?eNuH*-'
        'H8>S%6{g0yqP<<=%E!en$8qThf~)l*JV<GOb~67W+!J(=;aRG>1?1gV0Eb<LB^J#X)QtaRo7o}s^H?de~=5bX$ec'
        'rEitNE(PshH@1aWx)|LRaWlDiCdWaS{lK{<S75#^;sRposy`GXn4D!`FTL491_imS{uvWBR-'
        '|ZnZwg7^Pd`Jz&bVvZhf2N*4xvI|GU2TM3!I{>u<y<2|AmUvha--ADvSHdQSc}0qFf+T-'
        '<60<QUwBfF3yM_$KC3FFgQ+He@+u0Q)rJZJN%^nBBq0(!npr|X#+jxqLBrR?Z$m|Fw66%QIn{Eo?$Pu%&2r#Spp%'
        '9wc28-'
        '3DBGJd%=40gvMe|op;>SRf&_r8K~r7H1LXpes$h7SL0ze<I9g{us6Jwf5@hqA)xxlhj<S9W+oBCf_5B|A%%^?Piq'
        'FD^)nk)Kcu)l_$Ei~OaWZfkYs+;5>oY&@(G(QD8ug=ARWg%*D4AewO>AeU!G5<8?6=IbH+gGXV%yWN&H$;UIj5Ey'
        'b(T7v(2_K;c<kC+aPX?<Gt0OjmYJ<<pbcP>x)v?R&VIK9u-'
        '}~F+hb(_!q_u)?822R9POm6?4QKUBs5>S$GRykB}<%+)Ne#fNh3SB6&a_M3uwu&2!X^0OB?Z%fYSy&ZKA)~R{9HR'
        'NT!UoC7*3`@=-'
        'x>SEni$djwp`urKY=ahcjF%cvAr$L@C8NW?%8MO#j>N&YFydKynDIODQgib^G;hy~0|xtKsyhKg$K7@U8`uhI_L&'
        'FUr)u3@x5a&IyErhQOSM<9*7F5zE2UmNqOhDDK6D^gW?v0N{ilCd6Etd&09x$0P2F!H?W@?&$9XE~`vc1}9Xgbv`'
        'f7~ZTh2Op%OS3?YP2K=_$qvIb8ctesHX(>VO?cYdr54=^^8yJaHdr`qdgpVXe&_E+{NYx^$#huqhNK|>^O2J`j#)'
        'HI^3de{_F^v<I;XYR0G7AVd1&PRg4^Bw_Ss5r?4~nal*n{ORzj8RPOk{GAfiZG00x2LTT|va)QNb7i>B~;eS!RuU'
        'B9Zak7(yQJqWS23#6^JxC<+W{ilPrj3OK%qEmi{Bf~;Z&z#FiV{&LJ(G*2?VBW7$=E^{NqZW^S>*R&?p=GqOtLa*'
        'A~aW=2gjN9(gaw6@Kq`?*981exnXvHC0i_Cz~#A5Rhv+dcj8mKoxB|NcP+G9PS$_*Ipecz@*aZC}#rm1L^2^-'
        'LE%f(0Qa!G(xI#jUg1m~#Z5rKTrZho{cS8Y$msu=}M1T5fWi^u?tBh`025D^^ja>&<2y4U`uC>*8tWyB@e$>?{?#'
        'VI(!W6?|Z>&*Q$%Zxh@HbGl?E-'
        '$;1y8foF`qQ?YJ*e~Ou0xf%vMrt^c=#`Xp||36K+aQiQPl^G91kI+x9&xOTP8CQl!(saf10XJJ7|E$gRGUQRdy^}'
        '8SMthU4_RmcOqiejB;`a`(|^soF`YZZ)pz6KVh-jKB^u0%m%++!(YccD_yr`c@)lHVg-'
        'v97EAQ{i9Ka5fuwGEVCELOQ12MkDCK5JN0TscmJ&gLw%ANyO2B@XJYPJ|7DauRW{;Us3)e7n-'
        'PF%&48wtmm@$eF6k!~;ca0FAKtp_skAlPJIY1Jobb^gSBw{rjMb$TbvB!fNJlrCLTXd&`Rt2h~?zU*%5hnkvAiyZ'
        ')A3`wJ$Hm{uw&~J+spc^QMT0>fE71|Bo`^A!<0Xy?G7MRR4^6e=7)npZ03_q|@V?EMB8UpEl+W2piYQ{_AXY19WQ'
        '(fk^`3?ZUo|_2V}ECN%}Bl{kik7KS~(>7N+=A`&)JAZygyGknVd)5civHr-*g#KcHH$e;UI-=I=thQnd`P$4x`1-'
        'OQeZKdO!(5JM@oCekt!9CYL}so=}Up`Uj~HLUZiMf%T<)pPDp2G=hmz0?r5sIISl5fE?wa#-'
        'xFXvDKPY7AC~)Y;dQ&6CFHLbl0%%)gQQxR_0w%qtmzfjLwV@8aN~0tV{&aaIfApDZ(9!esde3U0~#$y(^LHzZnVj'
        'M~8;`GXbRr&&@E}MzhGw|Hz4;BAoSHs4$%Av`|1LF{*mANByg!rVSJf{QUyh&c4iSC#+NifnW{sN2nIys=%ylizg'
        'te49ZSA1on3c#w#?t<O@0hWbYPnwM2ZiE`ePs>aBx>Ubtj{Ay^(*1p<|YQz})Rezyu5Lg9hBpxnj1t3dtB^N?fHR'
        'm32YXDv;;t!n8*5|gBfEi{Xo6jPm<z_zUkm&X8ljLyM7VB-pZtbNW^G<u31ESqWr*x0@9@-'
        'VYwyp+UNj)#AF9LH&I9m^sT4FjNr%_RHG#+DpK?@Q2PB&A`t`r?37gW-'
        'o48mKB{!a;q@nEkK}q9|wH+c!AHVLaW%<lsu<*p+`HHA=+mzz--e^Kjjq>TRbdz+nPuLt}r<5#1HgnOqCGyipgBn'
        'zBi7Ov)lBGTnw>a{7}3F5~TYE*J<P;+~Ka0e@*c(C|Z1r71}&k4g#M2`nhYDPh(TUgE;5l?jLs5RiJM$|F?K7a6y'
        ';o320?B7gWQknere)UNatbLX@Yiz}l71KhY$N-'
        'gBpj#(wf`t?7~98anm<VyV;!az>{im^$+J?tHkF^g!(BOE#@qm3g@V8T2r@^{3=Y3I&2{3+Y25PKQ$B@Fj?u(7e)'
        'jf3s6&Mf7w3v@Bj$frxo=)C|q>va%aE@W)GHh}KkTayn%K8Q&78fJx26GBe8oyw4TG(Ww2mrv<Wfs6M?gYZSK{Nm'
        '<Q`eY!zp4Ehl-hDQ6wbPJ37MvIOZdi^;!eOQ(%7>WEdcAWQ!9||1+<INr_#csy>Vl=NKbc)LGA#H@+oWFtV%K7?('
        'a^RN#~@-'
        'C#q1Vs3=EE6{fe0^v^WgM7%fqY+Q2a5h(PjY#u&^K9ft!zhdhTqg50iD+@?x3&ktBR>Udw6IPx06B@m?SV$Kt9Au'
        'Jt=s%E;+JGn>dtX0K~C~-#&n$^ce(TQj%9pAJFUMI`Q^|VMkSs3se9*XW4Y;z@%U6g-'
        'K1#a1*Z*q}rAkBpBFZY445+v$tdI<yDS9PC4$AHyePh~3bx<cUVY=Imtf#3(}kbz#;D#i!4J<@@3B^{?~AhNacoy'
        'J<Lsim~H#?3ub#mZt`ssY&X;hL1ok?Zc6dn*sxssTF{^%$Un{?pcY;I6#DVmH_N=JsZfU@0a{u~zbZGH{K5FqGsT'
        'K|gSzR`kiCf#3H7Kp$XR3aDY6t2+W_-'
        'g6D%pfB74xbYN@1hE5k90^!M>`1YO+?qNy)TU2@`DSHXOqD3gY?c5}ip%~4MKBb#Xp_FVDZ$WgXDnRYbn7ZGd2dL'
        'KE4Evd!;S64M4`Gvp@ts=q&DnfK(a7nN43so+j2aS0wUGQ-Rh?THkwv$%Q1Z~At}L)TN^3Rp+I!^NQpi~|LeBc{8'
        'IMJKjjb*1|<GsWr`+HCjRmwUdeIz<KkWecW@>|zeaED(HL0VJkl5dX;8GF&ypkjx7gf0>xx}z^YCb?^V`I~`Ubx}'
        '+7FCp$mTb3R9aaF+S{dNqi2;CAsgnw<3$V9G}Rr7o%waSYlIV<#DdJm2D{*~8*>?#Gk8W%@ZgQfp@x7G7_)ONapG'
        '_z-ALC%WZ34$Xe@YHop#WE@RLGX-7dw(bTO&V7K-}dFjvsCZYSQl)!C$Sx4OBbDhHr`ke>A?m-'
        '{KcLYMa{?L}&u%oB@4cm!c^Ree15LVI^)xj2~Evw(;|66GX>Sse`ou*>)j+j-'
        'cvNP%PB)VpA^ups<0&%ffjbu3~elEJ$-'
        '%W?9G14wyw2D)_Y&d1;Lg;hv6Y86S7<I;e1X<RXE3vn!}R@*``UMlyjbFgn>xnrXJyD?@^0J=g8gcx$NI1M<?xFx'
        'I6o_Zk#hT~6vP>jbo#rr1q&R#^JSM5sWxmhnzmZyXW+^xM{pfjY0rLir)q}#U9JMY23@{*~1EkIZN5|AR+$|Kr<&'
        '(uu*%3<sQXKKKa>R&uz^3(I8npN-LI50(|deq^nqaD3%kQ4am@aS1|8dn2L|9837U}Gc$%8lkTp;v|nZ9pdr!*bZ'
        'GDy<B0!dccjToQIS-%d3Nw0;hHtNPE4$xq|ya~T6n*-'
        'Dl<+$2BF<Q821FjogM6L(<?bUF$uBvk2B!wRUitU%q%e`!#Gj3}_SmO|v}`5MK%{`F!#k+KP+>@{9V1_rIonW1()'
        '$Xed!IBtAQNW3v0$=xv@O#Cr{NttJXt3#emO6*RI757c(Y<!5G3>XivR>%E`cR=o#f3C{Bx2d=MtXL1;bzoqN6a%'
        '1jO~rJE*R_HUb8$AxoA(B}hh&I*G=`701jGdBX!`IVkj3plhP^-'
        '$Y<FP1^m_O{$~KPN#ejQE2i&7hBsmHoBL~o9bfqAz`GHuEZl&T_Ekq7gQ1&?j%7q6*xtHaQA-'
        '9@2$e=X`S1GmWr+Nd%ma((g!X1miLFo$g_5w^4vzP_GoAi#HLn#5qI<*HbeADT;^)26`JkYCAT-Ax}t;xnE5Fbc<'
        'X=Q!W-*z<O4=`~;{E9E)Ov@&|%-'
        'EtA#X4xpLtyEdwpD$yktKb5$7=|$>q;nRAJ`T4wI+cLUpD2jw?2ZA&bU)Ika=}P`1m(<^ki6rZ@}Rj=ukU*F}_C-'
        'l3#Djh#@B?N)iC*gX5s*ajHB+jbhTdK8S^?n~D>wf3!dG*oIR=WSJclXLPs|Mk)c$9}>)_7xHLF76>;1hn)0bBae'
        'B;;~pjMq!;cZmD}00wu_LMi;`+?0^VPD8sI?#s<q8$LbO4jM$B^lwPsJ`q%9bAsfZt!pybZWP>SL!gY??D0)w2cf'
        'n@3)jB<8A<syuS=U2H3gE?l>XhvZ&xUjhuORl^0gANy<%rT)y>INBRI!V+ob@RDagmVa`$Y+2VY*ed9QtlR3jqY!'
        'lbgGsn-ESHOX>^#a8%W12JsVuBxxj-'
        '#fwEF8CQy~mlU;ydi+fqDZ;`u5dO4<l6?WkhuE_`v*iwOz!*E5|cfx+7D~-'
        '#JyINx#IUQ0<Tz%Z>t1MS_a5=myw7{a77N2Bh;t`(ja*~^;NCs**)wFJn8e2zXz%H}eh&!ZLG#U3=odZgL%2T!<E'
        'L8or_*5n`jDUncce9B1Hk3A*U~T<`1pNsImjSR}8)Ln_(pHgXTC}QPOu<e48^z1@2I1Kl=<pG{=l{W}eqg|Vzf=8'
        'ysCwF|egKens^5LRWBtxM*Y6&2us?MQzV5;Ph$tVCqN9eTWo>0L$ZHe<#h*qmRAfO@AT)#8w9wavr%&XszpFCgo('
        'gpjX$rfSSDYIw6g3*L%aZzslkq=rtfwIj3Ptz#FbQ8LcyMqhV-HYCVq?0O848V^e+`8$-'
        'MZElLJ}#@L!Cy?8iw63Lvam`A0cp0GwKgw24ckjGAvhY=dceFZ$*jaFP5V2g`GtXy1}MISD5Cbz3Ie)4|5hek6Rd'
        'fqDG>SN8*o1{5CxL_)eR^k|%=<GTmj?7N4QAiE&MH89C}Wxm=^%6ff?v{9%@buAl_;9&|Sjiytu8G$GjkM{ikR6d'
        '<{uk6#K3qKB-lgMI>}GBLQMJHd$#3Qz1^6DMl#YQ{9aL3*o2=vH*TDW9sSh@QYqx&qaQyP5Sy$m+khXHa|CV_Tm)'
        'v+bF(tRoHz16HCL+&iSy=SsH-'
        '0#j3nLdrN*6h*(0juuPfSnWHnwwZwJaZ+nrbp@z&m9IJId84~zsgu|(CjrsDexk1(W}$mgyv46l&A;D`miPe+@k('
        'A1w~MX>f8K7}{7{&uO9rmMFbC!IAj6^iFZ1NiM4!!tIx^^t+=kJJgm1xc`&!psx-'
        'lsOUL>c6t7SAcj-v~gSl|N5<Xks_9wuB_L7Mf=9-7MRu>ZPv4gwHl_YR-oO63KOMR_!7Z&Hh-oqoXb8H1DozXmIQ'
        'kMAob|HS*vo;W@Y*9`Ew#J3;n<hkI77w)wjg2yWKb+Or3$MiWq00mfP&RQE3z9`9vBTYr&>6=ry1pHDb-fCRTg%*'
        'd^D>BbSK*D1c9xLMBKyon)JY6Lu5-'
        'j5@xfxJ<Lh;X9+4~AOuJykG9pz~{%6r&Q3g>?|Pt?ad_7QAGFdgL@3`o{U<fb~+iAvT<Tbm&nJN=?6-'
        '*`AL22PyNVEm$E{&UERyLrIt$hni(i<LggL8*_d=>s&sW(eN7kvdffoW#)ROXoNy`rhCpj%WB?sbR#QaBSm*rH*-'
        'S;x8yNzYRfZ9hh%7V3oRYLfS19x#}L%gzAj3(Spt@%FIxN?BhJ&!xI$R1IQ@P$iWeRfs(&I89PJ@t^t>faYMk{=>'
        '?j()`3r1V<!YQb-_GzluW{z#U{haNB0!_!pD(i(iku|*gk?yfrB_Z_euE(Xfr(kVQ<7M39}=cdOkEVSOR*)@^E?3'
        'd)_qsv?+bo^wF%Tylw;0Y5^{c=C7tfA=B8@k_37zi`q@^5?ncfudWvw!6_%S6=&L>A&jHUhk?5hV=mFBN3+Tn3?%'
        'EoIz~hLd4B%Be%j#&-'
        '_FxhL+Vk$uJZ9~;i0fjNq6z0>U01vTma^}w;5eZa1yt_;b)fo1M3REi{BZ&7S2)|W3#zzWRE>;9t4NRYs|veifQX'
        'oaWzApaLCvykxW^FcdwZ<552y%Ki1h?4%@w&hIcGi8upcc+Z)7}jZyX9VWsQ9-'
        'BJit5I&(+>>`SHxxzFaZjlYHIC|md@3Q3P?ggy5*u@}pwHnQpCF^=VZnf{yIlMO+PiOJ~hTUcOD(iuPlI!48C<_9'
        '96<H7j_WF%q<F}K;vAGEJd-'
        '2Nr4rU#A$fhtk=_JC>rafb%C$cwZKI?_iTu3ee@(XCVYcQSa>5E2Ixj`D<Wv=M`Eh(AnUxcGfj~Mp`Y7SHN{xk@J'
        'qT7@;(C$VH8F2w?CKAeH3*)a~muvL}8QF<N@wP0s=^rw)j}C<WOuMaT-'
        '2j8hLB>Pt%>p9tG2@00AD6=ghy1Nvg<DJ^)>r)4NUl^J-HMCVIX-HO&$-'
        '%_*O91wLk_Dqb>>KV!wC9I4K(yi%oMm_x=>9KddeY3ARMa*$Pi5z5b!nxs;yxAPMTUeEN(DbS!6t67nXPnH}9KM{'
        'EdNl_xMAo0Rk+|og-sWl${JW!W629>+0szoVvt+gk&Z_wk1sk6+N9Y#JCcKSX%<!@yYD;lWl}xHUo<i;<6?N?xR{'
        '~+_Dhgd|9p*LU=y?WM@9Vs)x5}@V<@G1=OmpE>gl2Do{T)X7-'
        '#+QWgRtB!_t%fTaDEHReew4dwc{5GH>nZ>9KQzsh2msgl_|3qVyaB!rX&+T!=woL<2THb?i{O<M#(po+N>aVwa(M'
        '?x9^^`Q)dX-kohO(yOcg@PgT0U*dAd^ozu-rdO-dH&Z^u}Awb&*y@*O^1OioLX(V3OZy432mz6C5^Edw}Sjg^xrd'
        'glSmSu%VLvuUV)-H=dEQx2oZEs8O1F^{nt9L6B1}75z~;J@hh@J8L&#3ub-AC$^nzx^!t7X|IFyhJnXXWVAxPMkX'
        '`1HPd-n!!mTC`f*F60IT|>`-'
        '}ZZ4?9gLVyGLHVR%Qr$U|Ouy;CP6wgRJmNdFQ>i6MAN6C>gwf!3Q>$3!c3YG@L_coQh5eTRf#sB+rYR*Cn3nMGpp'
        'HfJp<Rd_?q=pL#ujsoR_71iFT3%t$zc6HR8si0vgKW}eS@my2n4asM~D52g!wU|q-'
        '?yTNW~u3&p`v5dXb6?`HW{}*Kb@Kbm8d8U=~QeUvTOjx?oSH$p@Tor<IC`*`Hxlv4v({$kdJO7t=1=7z}941^iOR'
        'f@aDSa-@m(Jw1Q0G<AWpa*xzlJ~E=b{Fm$%wVGZbtin-4hb-'
        '(x*(lt_JgO*>1JC@Y>HZqgLAL`efgJRc9!lN<8Z%I4oFI_Pir6;&&5Sk%>RkHVr^Haf{NEEwOH>h~ePWT~ZRgWkH'
        '~okzW^`y&IfgUi|lY{=>`5^UH~75Ez=gU?+hccdB~~M9{{?>7e~n`Vv7T!#RPC#~9nH%taGUWL9KOpSxq(>g49*;'
        'B9>l+kU5Khz7R1?<GYtuzVurNV`br_^voqdsp=m2(p|bFTc&)3tL#`q}<T06?M5!U9LxOp`>S2Y8m!h@#SzvFN#a'
        '%SOr;TPQV!pSv=dliVWGU=Po^7%APISG;Jx}NHO4~3WrMy0xu!Opy@P3-'
        'Zu(E&caGfhP`u7`Fka&$T}Pi=d1`1kLC!XOk%SH`4^v}yrR5r0-'
        'L4WT8*d%=LSjn<p`1#w3y*+n~rw2Yvqf<ce5p4hp4&^PbF=6SAGIv+>7Fy;&3Qv;We-'
        ';;N10Cs)(n`AFi}SJhlyJ!$XdHMiV9|QCVz?uG}^IZ7RgFgXq*ij-'
        '9^}lY9BEO3G8!ny`Bn4|*TMjy<AlqkaQ5#dQ}ldgibbqE;zeor(fu-o8^s-'
        'zDEJMkB5b@FXUGy_4q`LNI>oT@Bz&U^Kz|5AS|>ef}nY|L!m6A1;Td%Z!Q1VEh5v?{-mj8^Etjl~Ch-Wz**|I4ch'
        '$6e!Udlic%8mp5!GJxfe3PVryzXtox(Kf(7k+1AqU)Xhh?k76gt0PqbU_Jzt`_BpU++lfvnq_It!0LkNFXmho1bD'
        'u0iKTV-uAfot;dCMpSDus2lnT1MUquSGWI**O~K9v_!dk&T3Zwe`7-'
        '(=6<!3L>lg#h0dcS#U6&WaJYV_<L9$DF0Xx@xsZ<0$ogm4Q82IFpq#MhP&uoz)vR$`LHpngFNOul)h)GNGs`<3zm'
        '?Z7*;5Ys+N1(|?c^hKE3IZaTZ3{KcR_Oh|&OtGX#~@!>8p+(ev2u*JXzMTw$-0Yx-em3B`3#FcFF_xU6^xm#i8G&'
        '0l(JC2w~ZNOI+kwH_uQTh>Wiq?=9398S-JN>Z*NqXQCDhSONRVa`68{KBt^y=mNmp@#*zW8`?K6c}Zp+P`-'
        'Pm+Y!&e;<j9>m$<48V6(C{G9Tm0=tC9WV3}M~wXFzD$&b-'
        '|>kqS7Uc7zydQo0EP`hkM`i;ljYX9q}&vGG@h|l`1x=D754E|cTS53XHrckDy8`b<Ujl%cm+}qI2p``jcR|9=;X9'
        'cvjK^v&s(*?Vhy8d!r3|I<cv-'
        ';)@(el6CS%H+*2Z28mwdmc6ZQUc7aX<cLtTu#CQF0r;3P*q#tJV$ZDXuKDL_i53^tM#)<MRDen>PUIal@_Egz{`8'
        '<I{iNrbRQ<BtkT}rBx>G$gS=m&1ji(xM6ZFLX{K=h#G=;=^#E`e;YX*^xa$Wqx<{4GwP)#2_Wyn@s=Xu`^S1{mSJ'
        '6?8Bq3bI6Ybg>*UV%I%A(nAPTnu0H>PSz8~Jj*Qj{HRHs2a>}A+nR>55Sadn5PQImfHFKpVETtCIEC&Cu?@)Tupe'
        'hxfQ5(I*dxjmR*p$Hm6ey~CZ_Y{XYC!3K&ozVrs&fMVy~tGU<NY;ho3vPPaHeZG}E3p7O^S%y2+Rp912(>_ilD=b'
        'KpI2|2W3Uxx>snKe;J+Jhk<lq1k65lgHPrdwwqt*aQ<ys_cfW#1x|2(sD?xh0M%z26>IXo2Ut+z4>A|LKZJTGDw#'
        '*Cue@B8B!>G6(T?;ul*!C`K_iBD&Aj6VFg@t=CMZW;+zwI=A1h49;V5A0^Qp=7{3}n`OuBGVC~|>m}8Z2!N{0!+r'
        '#~K>bYi>vucu4Z^~FXR$(711v-0-yVSTt2E3v}9?=Mraet_5=?5Pp8jo(Z);~EK4#YVy-'
        'E4nY8rg3+X)Z8kD7YG8z()B~d8ajlzPKphs4TJf7AQKpZLFib;I4>b0vqi1kHk9;?SJfcj_(&A^GXwIX;C!q^$e!'
        'JbyI0M*Lclg>-n3PZ$DnVnv|Ay(wn9BB_KaSG;r}af)|bPc>ns{M>o0Y$m0t(YGfM9+-'
        ')19Df~N<QP_#397`ljwzk@bte@Czt7FQPpR54OVfNk3JoF%be{X+JPA-g^L%0En-'
        'G|l_%ChFvC;##B&FiEuZn^=oxp%8$Wb>mjf4_{(6mLK6s~m^i<_-'
        '?QhM(%Y{oQHb^L;W8`;|59zVR_L<LvJEP!wP>7VB)~JDEp~g@tEF2NsBP9gfycU^YsgZGCha&C3STOc%|6JAx`M6'
        'nl1c2q8Pk7Wg_v_UKIAB0YhG6!feQPMx|WZPPgDw{3@K56>e<UScp?#R+dQWKWqbz9i&Y8%UaH<C_0YGoVRNZ~2e'
        'W-MP%ea;aUpzG~SxWBy=G7I__K9loSNW9Z~X4cTn?f(AFT2N%DXhvmrChQN~%k^2UZV-'
        'a)GN{;#{AMgXgFh340b9V3-(pZsztZ|C5Ktd8J=;cL8$~_<%g0Fbu?Ee5u#NFK'
    ),
    '_portable_underwriter_d85916da56a6.reporting._underwriter_movement': (
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
    '_portable_underwriter_d85916da56a6.reporting._underwriter_html': (
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
    '_portable_underwriter_d85916da56a6.reporting._core': (
        'c-qx{+mhQxvfw+v0>gbFBsQhiNS<*Y(P589p5r){?Um$nPDBR-gC>jJut<Ogz*e_=dVXu)_scdfb-'
        '#dY+2h@vn6R2aRaRD3R^BQrF&>ZKG%K+=sl~o;%S~2ZUbf=0>_yg!UDNheeVMJ>W}B6pR+OvTlWy6xVwLUSKh?6Y'
        'n!1~f$K%n68WqKQ-|t&d6j`;!?payaO<z*G(MUeqmi-mCUpAYKkWk8trRwyi-'
        '0fiCY4%S1ychLSaO+jsm&;Arb)r)(^&{Ov^wn0V*FV00HPsKf$qokJR2Qnn4;q*L7ROZ2zp8Ji*|$|MfG8UAdcWP'
        '>W@VSvJN0N+)+=}b|KF_?5%5KMu@S|-UWxW&i%^QypI>}_`q`>{@mcxVOw)Gew7KfH8-'
        '+kC@I~<Y*AB65uEkc=y&7gR%HV%RUx}hC*P{3+s>`chtcr`O?xyt8EVb@k?8@6sQ?90?{1Xro*VPJF)y3h>SN~nS'
        '`RbeRiq~I%{rc7W;%{%i`D^jR_uqW?{@qkJc=zMmufKZry7=n9zj^1pd@W(VMULru`S7M(_TM&Lr`}3HWmUgkuYv'
        'Kr=eQHxM{fz~S54iE8wphMEYJd47eY7BN24Fw21u}df4dXeTyf9&c-=A^-'
        ';Sr*xD(d`s6&5W?Ay9nHeEknj9xXH{kHzT{m*g(6M*hr-'
        ')8^NJK%f`^zIk%q~6Vd<=VEqg&(`s>>Yp;9rWMs%C_p7`p3HJ!~Vd~1HDJZ4{yK!%eSxJ6n}mF^;bWB`yPn<U$5V'
        'usjYYp(59KhVV>O)C9H{^Wv8>JOdm7Dv+S9EXr_uyUra~$X2^H1-+cAm`)^*o^M-'
        'SNz<(Brq4^O#Zbi2PVItt^_|>MYwwXnI)6B3mHvEBqfmB<jyIkY?5Zu={yQTvnXTM$U+iOwW(6IaMeghKkT4X}D%'
        '(|;G*8bLOPs5)qn?3ALyUTq@*}*v*9y@l*u^wi|43+RYJn7UVj+l2BpSI0AiFyA8nq<`Eh$&yh8vmZ_8_x57G#Zf'
        '}%YHQSlZDUgwr$$UTOcRsF#3_7F<r*vw`JAAgg#!0IxDlj-FN**=yG){wRg58{hM6~kmAj1zZ5Ih)}t@A=A3{4|3'
        '%dE_w8QfDI~t%k&@N}|F!CN8;~KO{j|vd4YK7`Q!PP!t(!J$YGp+5nAo+i`i(dv*)+!`sg|g7Nt8m{>q>kaOPxcK'
        'v;d_H&0V=_@dLyBC}3p;sGyIcJ!2sS!tFa8e@_2!vu0O(ILoRUh2!ZndbVozsIxZJx+jIFUq5A0Ppjy@;5VxZrch'
        'n%Nh{h7p2-'
        'GUQ%1nHg7^Wp8{iCYg3j;@GzY6(x5b|&Vp`Z27fsUxF6FKOT5qZ)7&`&HX!8!JrV;zov(Fjrs;;*CZLt)aP0>}q2'
        'm^{2a3$7Y<pJCG1vKr8V$zAtI?qnNAWEO<EkG-cY6xD>=*J96wwvU-Nwv<9IhiN)Q=$;^bw;X?*?-L7w!q-'
        'X%ym<ri0uyKl>?2&vRQ}VS9O-'
        'qZU@G6AgQ6Xa+qqNIn99E4O1GFh$9sZKwLGOYp|>wejP@CKDRX;?KXz_hdC<TLle+l<v#4PJ+%0tsXAZ<Sjy|FL^'
        'EvSU^G9?4^b7-(i+3X>e*qeR8I$~zm?l<nMCQ?Q7D-uAe?aRfk@d)iQUU%kkc?+v7(nBd3f1xZsBiiP-mw!)-'
        'ww~lznxHPDm|HTdfvI23dN79{NJae{ZBv(9@Sfegj7e`7Z-8<TnV#kkul0DS3@F1XjNk;O+G-'
        'NrfqqA$QtI0fQ2S_CL}I*hW$&6B}5MQ>TZN5>*FN=#e=lG}aUaQ;s_?d;>ymL$4UCmugfO<tPKJ$9IU+y<YILCUJ'
        'ZTJkYE&&@3BK!kaq@@_w8L*y5a!A438YuP{<y3^_bUAsT=;ID?OkEPOXG#1P_<g%2=~cw$Fk9m0;I9f5D7Y?Pgnu'
        'LpFZEk4{`>2a6o!Yt;rb=l2sN<8PcwOSyTKP4(!&t9TIk*D@lK)ryz;{%h{r2R#-4I-'
        '4_X24?X05>?HnJ`FVY%of1R(u`ZtNcQ5RT}4Ic2GbkaWHH|=0Qy+>9tVvbrr}U2hdytOXwgsq`E~pBi<Nw)Y0594'
        '!a)^TBxN6=9i#5ccSfYwM`edPK<F!R<%J0eGk#2AJt!x9|R0n^-'
        '|Ho^zf<F&nKjxVT^LWVNbwU<gTWzI?s~i?M{Qm?YQ&699WFJj)eXs94&viEA&=$FcIwu{n&~1-'
        'j18<=oSwRj9G4)`m%$kM3#Sl|K{6lU4cEUy`pB{@Aeo}>#xo@B1x)JpsfMM{BAY-E1>vw3-'
        'X!$1B#m=(|HjSyCu`0Z9aAd7+7rT(<77xGR{}f>*}(`m@c<CM2`YEu)>h)>YEynmM-BMM~q+!f_`6a&aAML7gFrJ'
        '8JLEQKjRqySc?+l%+R~v?>2(_&StZP@9*r<Nm+t_*#$7XR^9OwA}44`0lT7Ui)*p$m2{2}f^8<!ooAUF=Hpj2b-'
        '^i&Xs;POnYge|LK_E~TLJ=_uyU3T51}z*6uX!l`44gH89dw=x8D~016Yu=?2cgmp|usSGW2MKbNaPVObm|!_p&40'
        '6{0KLkaOCAxtb2Ng(+5B@?%h5Y(zbQhdDmLzX`fEzr%JKh>R0}DC!Ql3J)?s({9Utc}1I!;Enb8NF~6u>elGQ3L1'
        'J=ZZ?zShg>#wUjn6@8k_+J!?fw-FjDdWXvHzta_4~Oj63H5!#tWgJ1E9D>22h;UU2iS+`?Wd)Ru}tH2X>ijxq2$j'
        'cpi&;KZCc=E7|o88AwQk|eLk#Yt#J5Sd&7M`i;vI>iD8mTpYVJPUz3sRJAhRj!^SM1d=gCMJoQ3D9C9-'
        '{j=y$se@kB<UO`bHR`Sb~_>NL$qZ;#PJxN9{J#$<bW<eYGse%FEdTKyK83~;7;pAKXJ!2oA47+!TBk4R(oZO%3-'
        'B&DRdXKEqxTS6Mkq*F^#nk!ADGswy_L?>Y++lDJ|T_+2afkVP-'
        '#p7c%Lfp;%3P*m<5O7}DB%`lNz565H7ub2JoS(#?hz5s0wj7^$+~Zh>l{0+M@#B?iol|B!U&T|d;%5f6A$W&gHhC'
        'rEf;VhIl{c0FtN%@^=yXz~ttC$gta_m5S7<<er;bQQfKQ5zt2OZW_p$~1`%C!y&f=9!s^R$MH+62ZLe3kra7^hsi'
        'LuHY@Qm-'
        '9m;QIRpmvJ0b58aWghZp=*ahDOcN4)|AKVc~Jw{Yb^E+VUf9zp2z?D2?o6eh}f?&eo6|FF@obs&AeO@{)~nFy5*L'
        'lP3Z!H4g-RNs?~JTgegSyKzx~Dldw$*#htv^cLg$11PNHIG|AkWEJf-'
        'o*tRQ71*t9vjOUoVc^<HDxLxuMOWkyB3f`}QIw)%S}h0mIj`<Cb0(XZ0W+~_m*(cf^i)H*qAp}@$BIgE^f>}U1~$'
        ')_{iUf6Z@^ZcO#Z3B0%z{G6C=(&V<vPv$#b|?r}0RMwyZD3#F(ebwna;_?mv5;8|n}g3}!kex}`q6;eC5Pf#hN9K'
        '5F6nU}C1e*T!Xq6wN$y<%j@a(83R6F<|hZ`)XZ*NR=m;APpNf-CU&(dOQjejP=Njf~M{ieKeGp<2|>c3n2E$cXU0'
        'UU~!pGnv0)cBRU&Q0|?zA{VM4c9So4sBp&V{h1{=k@i<)00<7|?tILViDJQ}4X~^34qhwH}l0b}TIf0(J#JF#7y^'
        'TvQ{}lfQH%6}!Ai6G57FNFq8(Y|Zz*F<Dfy?B1g}7OYU7z8RGogwZUhr&4d;4>dC@cD3mE9F)NR4w&r~_!W9d2Wq'
        'gBhbNjqpTvblm0|$5Q$5K(HcHSXou>o`7wW9QdA2y{mfJTN{O_-Thm332Sy2%){Doj#KP=eFk&l!#NAMO|vxoh9;'
        'N5L+6#anXKAo$5APBAjPVFZW&zDh+^#KcdR~nbu4|RoG{z`iD@2X2i+U+Y$wk7pV;`yotQh~(!7rbtmc+Pv3c57I'
        '=egxq?-'
        'rJVcMo5Tyyy^x6D=KLT3eTMOkxT$BCy+BXK%EZ0tm0acW@emX|V~)hP&_q++heZrYUqj0!piog|`)GQfekso~=gH'
        '#fr4Ed@wl)PfkQmw^e6E80`E3}>DZVWw<@OOzlJ!eE+x05+YEyPo&E?LnHsM&>kt!O*|wXJ;p;3j-'
        'b|PB78V#(&T0soOxic9H=jO{Cj1m!z|sTr_)}R;A3Gwm8R~HFd{7KRsKZ9&#Tdt_<a}ZT33`u-'
        'tFa)G~0<oj*0LY>c7#1xOT37MsvZ!w2r{a;LwZ&e8p|M;<L0;EkU1)7jIhg}sYuo;~q!XKQjHGmH9iV{mej<c5j`'
        'oTg|&pZO;hH-yD{4?=fSeGn7=U6*yPJ^w+3<?H}qd#uIC9pYB)u1k>eMTZ&MAljGR@cuGV@VHH8Z6rG@YwF}^$<2'
        'gFq1WPWcNNPpBdZ0Tnyn<yClayU+c2`rWMEWuei+y|Xz-DE!nBbSa^)W(EIeZrzckfq0%G7s`Vc$OVhV8qh^s#bC'
        'ZPTuLSBM7jD;T046U#N>AeNr`U`nLlmdqlv$DSef-'
        'E;pCsgNYMhsM5bVc86=BMK1^hM737f6XG3bM^O@M3C>pm<1NnsoudJF*id5mwNrVrOAe_MoCb9j5jm%tV2i;XuVj'
        '*?|v3sTR%0g7Z^(3ic+|nnWI`6(p7?GSiNN&VfPVC<r^Ta4M}sHDlpN-'
        'h=5$GXODEp_=rWm)tXUnETAylB9o;$y|lZvzRF!lRs_238b?jd|?Zo8fA?WI*<a0i0{D8r#6_}1z~3ArC@s4ItC}'
        '(J1UB%K=?EAP@M9T<Zix_GwrEoc)Gs0gRh+=+6vHXD_GMF$is6JG{fw9j5eGvMs|vqEtt?=lqd{F%?$4VJe!ebmj'
        'VnUCu0gZ-ZYP%jPrXWY*BAE_FUBeP>pP=tEO2M>vCIdZVxn4S&Kt$z!L!Z&@sCVHSC&w>ykKJjp#IZ`v(p?uqJM5'
        'LzNjt;banAhk7rNM=oPn6wOEJSU`+w)~NjK5DBdg@vP4wvRa+Z^MiD@dhs*egNSP}y|*oG0mad^s3ajX3^c2)2*`'
        '>+h`u%9ZDV@}CqVC1mVQFi9RWh_;TecnP_Zrx-GwPse6VuH`#JL3rW!PJen;9nq6NX)za#B;o!GCMqJ0K!<I%BTd'
        ')Wb6N03PYfe|D}<-'
        '0h<6n_*xCu5Qk;_}WDcib?l7MUlq227DyCqgxsmymAmlSm)U7?w(cj@GZCPWG)f?&z{!q;GS<M*u*#LN|6tM7!w_'
        '7+2ImfA>M8M<9#FXQ6x=?nh$B?M9D=dl|Xob~Vq3`?Cac`|>ll-'
        '+eqxWSc<~xvj&P#LxS(?yHTE&PyaZF&G8Fys`44+Ejhz<UR0ZH))WOIlxbnzP}1&M4W%=6F_?zc6yi>zpYl3D(m>'
        'yo_`FR^y%z{<>vBp>qrw%ndb)96R)M^s7msm>sSpX?d~RqROopqpgK~Jb0&k;ijNR2`YzP!S=-'
        'OfV(%VcmRlEZ<SH3^&9mH;V?mjF>1OVE=%*TZ;1I4b%8o?3DC^b7YSmwX*+Dl>t7}04H%GHwmiw*)QCfUh30DVL>'
        'h2g@m&w9u)$`iA!|U2NObC61<|@T8-g3K{fo_0xndJVJ1D+W)k$<Hc>X}1RLj|{uOA?fX&I7Nt-'
        'sfXp!cbfOQtjd@bTZyOJ;hVt-RV>M=LP@sx%}tzaYSX2dy<kevHztO>&S(35C$E6ikh1K^8)|#IsNDJ1fq$@e7Z_'
        '8hUtOBFAKuG<Mb1=1X4HT*@p^Vy^XCaZbt-PW%NkhMmBdL8Qn7LwoQW@?qc6nwdlGNg>&{diF5Wrw6)kM7S04%t6'
        '(sR86P3PzY>OJ(Wojj4dPw3xfWoV0A2PQk?rYl3J=gauvRvlZ;eO1fa^*oSA&s6;)8Pkn+rS+`sRPW&AMf|5i`8G'
        '4s_!dFcd~z!195SxatDoC@ea{o6ELZfdO~jR2V?QWbeQKon*=lp3vaE%o8T*-'
        '~XNgQq7W~qfFa?vXVK(b+>QHn!~f9HEiOhoAKP&6`l`L{0dQ$mKa_%?HfAp!9=QZrU-AvK(!8I&EZC2$fWdzpxy7'
        '$qi3L%w9@Hd%e5++<Q(6}T?6+ZA(uSaDt9M1crZiA10ObX^T<&IZzW#U!mQ|=&Ft=H(Ei)<W}-'
        '$T6I*us`5Elqpa1m2!Pf{A7wIW({AaY?|MWrvV3sL}T>!`gVVxjY^mR{Xe|-'
        'MAJo|w6NKwworPb!Y@ETf({L}%Qn+)PFvZo6Jkt~LK!qBOe9Eh9)oHRHR6QUc~9x&}^(+pgS&lq>Ji{>c2palbcL'
        '_IxekC9FQ@+g~}01>3&q)cwgz0j-'
        '3RGT3XG<?)gKr{~r##UOiMO2YD#uMuBXN<W@$pka9grhU(Df&k^bC~nz`p}Gga~T4_=Hzg>m=mgi`X9+=(^xU~d+'
        '=P_=w$QkvEN0fMcN)8jXq^SmT3|oMLX#vdrmQC4+|R;YWLW0H!`=CL1K`(A^GL`)6YMf(l9Shfg_$<)KrpSb*WCA'
        'HqdWZE?umr9^A*U2)y~r4pyY#u2*1sG?(BX=#7Mp=L9_1$GSUFpzCI3hexRs&P~@RCYIq0zzW4iych)e{B!|C<3F'
        'D+q<E+<x1}P1VZSE^hCF3hRr^jZ0VsDQmOIgt3?JiSE=9GOJg3z*&{alivtD<icT<y282VI=@)$t{`S()!QAcsJ!'
        '`=3RPZ*>)RG^<Y!`gkH%;jH})3D}{0BclsRGRu4kIyH(WZYv)No?~mHkaNJD{K_wyf!wZb2*hWGbe{!^1_Fxd(I{'
        '1xyWV0<4Ncs*X&HL)c?r-Lx^1_>Slj=CGXzwN&jB!No7Sa<Z^Suvf-Bo6F06!O(hp3v@%<ZkKn`u23tE@uY*<mz;'
        '}LdMj3<H?w3~?@PlY)_6cNS4<MIVV(96!=W-'
        'z&LQD4y91lr|sYg?b3#Zvetou3b{+7}fP*?F8188+yQJMoM?a$np1UhQ%93&{5fIIM|cjTyHFnl9y)Du@7Xg{Cxg'
        'pWG^f9UyS%9%`O1sI3XiU>#>l*2eTD!}~7r<wY%Jjx)na;J3o>q$l&(P+A%Q{==^sYF~Jm^#sVm(BdC=A$5XbLab'
        '!*%-'
        'Y><1gOpi&S*KT~?I(jr@Of=TlBURYC%<mm|OEP2S!AdawUq7!#N7y25vna@;pfE9zfhKCwL$aY7{uqlfsh2mDuFl'
        'OXh<bF`dO@9Aj3gB@R%ShL{&i^)*l;DOns!ru=GecwBY2h4HB5j=MId|0vAg{YTT+p_&IcKPc#-'
        'Wphn!>{78tmF!$Sji&%Q7rU-'
        '2WO9C9Hf~EoK+qjh|0td(T~xUc&Rmb<>Wg=ucXS_6EIi>TDukOVoL=FCai{@+b1H*bdblRUJV<blC(?lo(HS0$WV'
        '5L9er6xj{OBXp_WOq>CvW%0EwhB%Uxh{LgzC3M$f={2UaF!=se~@ayFqi&E@HCBA=YdSN;J192()D{v`sNV9%liV'
        '-PV)0SX$a7$I{r#z3hYtaDb2S`BrQxrf*TL^dh1c$Etmwrx>PiOH(M{FwQe6Hdn{mFRw~e&Xeh*ZZw#%f4x~<FY-'
        'SXv6~ABk`@b^w0zW+H(0p5UXjZPcj>t`TK-Cd@I#Ol<M5)qEid~68-'
        '<@k()Ev+M5W*<Iww}uIJcb;k=i~@f*jYQ~&e*HG&u(;XZa|!NC>m#g!eX(=_`dZqU!H9W&V5e)PvUz~>0?*>ewKo'
        '?=?vU2lM?UN;0UsS(NHiQFG^DmBFmG)eIZ!-8vt`Rx-'
        'c%bIa>U1pBu1>tWn$3;<Iwu0&_eX^)MCTlVk=?3Q<t4rvx@V6CKHLUTJltP|3ZM^niRb)L`msD_%vUhbY4#}=KdL'
        'e+NHol_*Vm!#6Je6+9L!!TK5}VyB`%=koHwu%FqDdiMSN%TPTaK!vl-'
        '5xeHr^<zseyaq;a~#}V$2#Zvqfzj@&j``2A+p&9OSY|gK{vk_0JGMaN(P6RZA)CPycaj4v+OnGSukOk>b72q9IIc'
        'uxQFkz>(BODSx>+tZ2q^Q|%@!bS%O=$((Uc6kF$xxL{~g514xT)bx>J0!+Ue%}r2fAi&W1nCNOOR*Hd>1xhE-'
        'j)Th;hWTb)tR6#y2N@fc;_W1;MohTaskf7dEd>lkimd#6B^=KI`E!{+Gv@0u@Puy3z^oRLKqEr3Gxc3a@@no;kcL'
        'xKDO@!5$YFI%=e&&$q@kJ_9Fc=+2sd@YA?c(8m;7P&0g}XKg7`(>fU;JcLI(?uAo3A8g2$xp+B7?qvg?z~D?LX0t'
        'FUw$hTUwv>tGmD$@V}oKQ0v=7)ES3f8qfdgX2}f0(}+D{)=Y*`Ev3TlKVa>vCofagsTH{h%A{GqjDCBGQ;c#N97>'
        'w+UmMo-'
        'g3cu^s+xK|KMPisNn!{yi{(&Up7{oN(IxkqbY_<?vXr^NZzv42Aecp^?@57CE<N{7?*GSO`@~Q>D~k=_a`Cx{sFy'
        'pnkQo!&bay<!0!QGeW02FExd8qk+_x@gp~x+RrTb6iEj}EC*?BE;tnrsom^ELkzriOI{GXxfbCgaN)nk4e1-'
        '&sh+O7&`1kw(fzK5UXt)K_TV`Is0~EiE$RZ~1Foi_wHt?kDrbePoEW}=>mdy(2Q&As;lt)YGA%dn0-'
        'lmnM5~z<p&Qui(xWU}7%h+QciM)-A^`ot`0=pOD!t<p&fP*mj-xCLxe@g(5uHtTnOZe_wLt89t4@47#ZoWPM(&H4'
        'Q)GYX3w?X&i+#Qr;u!e0L5nU7mvJd0~drZSf#Z+w}m&K=Y>7hMXNS%;3&VtKZqANfX70lP>iWE*Fa%;<}_SLe4O*'
        'bN;i05d|6)&^5a1hQl>RqizNmE3y)LZOoS0<G=Q6;&9H(_7*DjD}O&m!&7c`kTAn-'
        '1oodv)5b?))pyxPJG$XnZ|5QD5GFt?S7jp&oy`L|s*%jm{LqbGkeoo^V3=es-'
        'B{1K&9XkIt+FHR$6&0oPJ0x2XH<9x+-'
        '~2CU^!2a8t3pYY*T=BrVI2>nJkqhc7_SJ7(s>RO1k>y3TNokhc1hF1@iu`*pWy$<#PIUVJ3zn;jR+zpARUu(f_9K'
        'GHiJ3orr5U$0na+n5AoM5YgqP^^|<_6W6+oH%=T->VqHm>W!?36_MVTdP@*SjQaKRlB#XZb<9$-'
        'QX84D!2EBDFn#OIjScPh<38z4v<)C7x0b1A(sZ>|4lEJS2h_o|kOjgiPHQPv3iUMm*+*+wZ(yp>)Lc(n~}KsFRpr'
        'bBX7~nn;<(hKOq)hbKO1Kzpu{IbMp|KJ#nA1Z%c4SxUTU>&uC2ez;0YV?a`Y5+q~jd2W(S6`)U9$kAuAmU=gNI<>'
        'UHG()zsR@*Nowy%{hsW*&-K8<oq9z(=8TO`;oZA}wrEWM&(S+<aQIGA;=$+Ada!L}y%WuQ}JQ0$3^{(-'
        'wEHp(#_9&}gGjnoPW6iPHkE*~I=9`aM^;i#2>7zU*XL3AztU0GqyAf?-'
        'ZUb9mA)M<5Ll?=DCFV#6l?DeCmE9zBRBaQulIxPbxQzT9hFSVj;HBqi~M9L`X&7wHuR^ybduFJYFIhCA2p8JHrTy'
        '6QvCE&u%i=XOjci6G&hQ`E<K$kO~^6iYz_4SOfVhP8&<uA^_ADOVegF$4A0_#{4h~+KbJ5tV{ZDQBcU2$jV`8;Hq'
        'QOMUog?cZNsTiTll_YX_f*9!>ES{G;aaquLn#_r9E+=*dht)av!}%0iOhk^OLp(TL@L{&kf#kx#eV&J378x)u%o^'
        '}Mid7nMmMKpyQyPH3RC%PJ2Z}Nz{7{t1!*v>t9)s+1Qn-$Xd|hwUe3Y~xX%1A<*&}K!BIVeS!5ep!vMx8bWRto#-'
        '2-5E@v5o7Ie57?%0E`IgYnrhw@&4GY@sLVcLTTWK&q`w6qWmD5YoUdlh=%;S)s*m8e9|tro-6<O=3#Ht$5-'
        '}t;BuWN0083cXsV!$Hg55XYV6Zw+a>~A>oIGu)m*(E+m#B66%_<3+%_IwN;&(Q2IV|da70(iyC7*(M6gJc8`h~*C'
        ')p0MR&`&?+4-M$+!3E%h~ppPquofD6sp5g7ui5Buwt^kG=SPKOl2tnLte6k<w>Ik1lQ}Bu``|Tv-89-'
        'i(w#%Z`s6WSv~=?w2-'
        '%D8FU6c9!d+;W06Cw!~}JHwE48AIowdei~?yp~7Ysw>If1;p!u5(UL%ixm^lcJS0=&;FCp87wppAWKjdEL#~c=LL'
        'lm9^8g?}+9-'
        'hz^1nF)8nR2y(|F0l%j6(K2y4X?%cDrNF8Gtkb{{W_J^(@{6P<bN5a_f;uI|Deg4mYz2mBHU>#|GHaP>e{EQnGt@'
        'b={3BDmPZRHKK`CsA7$dbX>o&_D;Ju!Eqhz{rW1alK{FR=kgGF#CT`js+apkb|Iyh!C+DY-OQYV-Rn(QiJBlV)Ss'
        'Gb2KB{xm#xkG_Za86Q#7YL5}4y&-Bnix&fnOK+C2lIb?GjwmXux$HSM~U-n^(W^oW!ez1kw7=aHn-'
        '1UC&?Qv$yAwyzdgWVqp-)U;eN&&-Wx7*^#Xn`H&J}^YYS+o%hS;9SFP2+3@2M3ToyduZ=hGv9aTW!nsR+j-'
        'kh`)!N=g4y1j-'
        '!_#9T#{oULjp0n8lWS03j!=<zn=5o<{d9688BhT7FsZS9a!Jz%1x;1;RAw<4<yEAZG{%zrnGxfr*WLK09uv2eFRM'
        '^#Fa`nI4C7WUj}MNZ&kT1VKOTRd)38S+&l%Set$JG^nI$w|PLWt<w+0iA5UJ&Gy?NW|s#7g$!5g-'
        'zsjMJUECSAcVr+=Yy~X?bo^4wbS9v>7r~GDD{w!);4C8mX$1sJN`!Y!4z2x9j?Y6zlS5zb<jbm_*qYfEce;Eyl(d'
        'GFnC7$<uTxWs-_3%4iW$}t2g@&KUe2QxSeBzh^(E*YTsV){phgRp67NZP0_KlC!v|Kp~f-'
        'st}W|vGJNPsCrlCdpC$T+LJB&E4->spG6d<9c$I+UUjRtkv;KkJpk7WJ)Zx)zOpQ(-'
        '5*LiPKv?Nk{e6grx_7nRwav9k_D`GRgYmC{t6`r|)>unrV--F*Y<C<mk;ksFGdR}+UR{hQlGfdHbZme%3N4Tm(|='
        'jmG;Z}CB5TAY(-'
        'o5saigmWIA~@o&;z|l!rQ&LCR1v=u#~34sQhQ5W$fg9d=@=WpC0wKyv`gl_;04kgCp;HA{NO=U0RHK%u$V@4S#cI'
        'ZSKX6rU6e51a;gT(z*LH-Hmdj=ezV8$71HGg4F7+m*-'
        'T?J&l{btt1246RHP&=6%6hm6lGVL30`A;wM~CJhPTzp4<zb6X;0jTvuB*0OV)rVH6r1{vG(ny!xzhaG&Q6zxnz>+'
        '~#LCY)n>T)Zi7Er|=W8BVhWgY3b3B5ZOOC4<#J(JUWmw#v%>d`Yc%h*{6=KP?ZnDqx7(9c?3K}au6D4#sz?6`A)6'
        ';WR)&@oO|*#3Kz?K{{z#2;J=vdLtko^P<?yb9#hw8k>7RX5u+baa&AcjQ*cwoccA^7mQZr<KA5ioIhPOKpxm(TDt'
        'LDV?5&f$Dd03R=`!+Tsdx;fzE+N<pSR^D&~(3|AAA5Ywd+!^*3?S$SIvr=a6x<X6&NGxrt)t!HwvAq&lLsb!f;ic'
        '<}*X9xHA-_;?S<ga99e?jOD_z+1-'
        'N4BzD%PrX&jPOx5RQSyFkooZ!*9+kD}F)kTU*LlSJkIN+xo&!%b>7Bf6#l;7U;F4(zCqLylrDx`GIm6YhRpV!E#!'
        '|#=Eq7pOmG}Egem6qh&vI5(QNKkYFzic;=*D>{TWaz4XO@Av!cJK5bcP+-'
        '$uVc7$%TtdslmEn4R^M~vHs@n{FkYx)r}&*=E#Zg<xB;;%kL6014$K~)Lx(r4#Wllzs^8!`%EW3%7)dm&({000vY'
        '$66Ss&77Pt`EGA0i)M1)J%z0hx>+crW#QI`!34(wJK?l*CGbw!V0a0vnSE`Ner?Wi3GDPs00`sCU6SfR)Id8TvHi'
        '&r#zjxqpvPudyTr?iOsA`sitdlJc+PyFQ3d%(o5HRV(u?Z0?`uc~A#*Zj2v^UhaJ|)uy+)5N%LkwHP9yuTEnTN8I'
        'HYcep6q4kZT93!?_~hb%kE!`=IY)2!=V;g@-W2v^<4E&_h=KW3othfbXDt3I#_t3CEFwo@=E!u}zz-'
        '&kvO1H=i)VaV-BV|DG_6e?ATylyUrMNQq`jYK^F1HIspsYMn)(&}7)V&Xx3;I4&Y3I%E!e&HfIv^_hokKrPJDb;3'
        'I72yqB>y1+yb>v`jV-'
        '^4Nxp(A;R`}Z+YBflMMsXf5EA95`9ZtU&p7f`OG3oCDQ{J!w#=kva{O|xUogCtm)80brPwQ>)=@ILlTyTf@Vv8x<'
        '%NhF9SU7iFpnS;ePkoG4qeNe;B`&0>d+jaT#(@3K=6ri3$8<W2XmfsL<EB2I<Y4A~J!u@M8mz+MKl9pa$tT<D-'
        'x0VxUDrqpqBv4lt6d;v*eE=D4ui~C4RCKG4FG2RFcOk(Maki4yEQ+IeOt|4GrN!dD)}ucRd+LD|D~2ZW&kE_J89i'
        '|QbRs9MX^G@q+4t?hgwlDQMtTv=%%|OtgreF@{=6E6L95C)$s*(l$@5h@!GXQ)^=q39U5cJtWf?MSl<z%*}Ce{@-'
        'p~(od}%i^))w7BLaq{h{jpRZ&e@ss5SU>S?ypT&BA{(3IrH-CAwu>?Wi5PTH>b9!s~%!Oyb;k02a%-'
        'gJePLi?uzugK~uC;wKzoHXi=M&d}<Dt<hIhpbb+N1)U@uyT0Q)xy1L`|NdW@gp^s32ABS==wN|5yOjN59&lxRgb9'
        '8)h=sTQN1&1BfN(Vajt?JQw;|{$f1O<|;CPfe+W<y9$T+`d-'
        'OJ^2zuj+2s%V8JHD#+$U409lEIlgW4f4S2Xq#}^pB~uyUmZERxAyVj)61P22G<8ai9i*XKrvxG9j=Sn{lBu_`s+)'
        'DPUHO4r54*=#on(5ni-'
        '(O>#e)0z404t`PUD@$L9duH`yc24)3oj3O$tdFirRyTY)5JSr8Ncid|TY@N)iY?g+ht)cX3n;?Sl&v|-'
        ')#I?i~W5ps^JacJ|ZmEv`gl{w7@)kxC!#c+!rLDC`9?D#4>N5-'
        'Ka9>0s8=Um(taygIw2IZ%Qm&q29W}dyUrlO?oYrTzp*22Xb_@bBB?<X4k-'
        'F@!Ge3Ow<=aA5}`OU6BI@1QUd7t8?ax;Zsii+1M4#e1=8^@4KXYykSiJ<E1R`+(ql8Ub>czO1PU-'
        'K*oT8_e7WJs73xm?OaP;%f8?R|DtA|e!y>MJ|~FcT=3Emo1>4~3jXMA+jDxAE+!2EXYB5>KowbYq`GSHej%uA$_)'
        '{C8oK-TGXLe9|FZthWY$-'
        'o13fUUJg9;!C}at(WH9GXA9(UydR_ssnt6Fmtoo(8>n$fC4UDK+3d~uM*J$yJaj|uxa{21|=Mz>Uqr1cVPN{zOn*'
        'Mh|O*u#vAY>kwtkU!fOP0OoxgiKeNjm%EDD9o<pC(f@XGY9eYSUKv<+C8nlVxtwMu3SI~WKpC3eHNh*w=0I$N%K6'
        'vY3VGz73HKHRc0hhQf>;<4F#i}YV>xRC{64)wHl!I>ZoXS+|<bA?0{l-Ri4e$`b^SiOq<M?PsKc-'
        'T}tB3UCo_Br8I>p9`U@h!`U)k26Tch+-@BCPcEfSO@I7(Y5#e))xM`+%CU0#>fh7&ft{0uC=ZAH+3<1=70Yp;F-'
        'WAEwNFr3CoUPlYmFhR~#C9;}DtOgs?KmfR5G>e)uu-'
        '$?SnDRqhXn|UUG@(9(E@OQ6!3|QC0OFx%jTvko1h*`A>J#PYgxRwh$B%-o&xP?HfTA@DJhpx*Hon+bn-yn`uGnlS'
        '*wbJNtXo<xJ7vK!HdSHwS9EOtefytQ*);1CmbV|=)^Z~{CsJ{UN?*ro&C_QRfrt95GnrBHL#e)^dep|0r%&>j$?a'
        'I+39aaQdu-M5Ex)2qq2$rZ%{U$P;72T678$A8NmBi2zJ~FZmg0MU<98xC5;T6x`6jBS^MBzh{8*&etWVH40n-'
        'iK7~^0{zmoh+W&`OvGCfU6XZrpmeuP#c3V5i$fXh=z%RR>i7aHEz(#t9Mk=bbL)-'
        'il!MxCEYXED{>`g>{}l3Q#vcYc{hijn5(5A(`fxw-e-`Y10?aPHH^F0HS^wDXLK@``_ZlUBtC;}l4gw6F;g-'
        'g=yRCz+|@jG7RhMVzDuU&@{zS4qY{ke(k*0u8+?pS-'
        'gHH*426AlkMC6>|$Z+Ha>}$w|)140`DY>N5gX)@JT*46Re=m`1cg*U}aGlaZY&1&m--yi$E(FWudT&$axqN49I$I'
        'Lg+jsWV^im_CjxK5QWbQw$VpXern=13ioHKB(~<sTn);$vAW0C|T|$_1%OAEwFoI=;0sVbZX?{<$Mx>^3S3-'
        'C9KV4EN*@{L;6fHw||%>Hs$B`51YBHY`=LlPcmSN-'
        '2U($A8pMC0$;H?2QYsN(U6;ufJ+OCW)!jC6H}63V4jmfF|iDd91ARz^=jXVc9N^5ndn4XZVJxadR^VjD3JmDR++?'
        'jH=bSf+YN;~;E(bB(3iukLHYY(qNc=Z0)aw<K*%y~ayst!F@ft!P`qW}ZR=oeKu>xx^%IYb6pi(<-'
        'yaj3WejXqjzK-d^tzf!0ZV})4XY$8sa1bO4C1QnC+f!umn(BqA=Sc1-'
        'Kv_P@ASticl)1P<=UMvrG1a#8nlV}YAa@S^KtUy`&ao4^y3=8v)WJadAAeG`M3jryvD@&0E=DmA-'
        'aV4!fPCClUT<C(M%m_LU)6b`imdle*c$mU%x5dy?*o6ckjP>_3nI-IqU*fc05KqF{y&-Q-'
        '`NpM!}?Y@guvHkzE1r+ltb?2Qj-6wm@*`nTNGOZT-'
        'n<&!`woqVrG$%9fa8=e~md#+i}r4Z1fRtT%QKOcK;(sj!Pxg~NFlQ?Sv`#|EUa!FV+Toi#I;?JOMB(@;fNdOm94T'
        'JK}EOOCI3n3|VHEv#er>&#Ur84kbuOAF(EY+Not5;<;7--'
        ')qT+kDaiK_<WO@JY*T0ShVeiAKwjhRYGgi#`&H+*XT_&#=Ye>uqi+=Ock((|-'
        'Ze+*^i;JP4R=Kdh>DB2NWmMKo~(yRv9LSh>}*6FKVzxuY#+ptObc8O{RDiissxB-?-*1Iq(858iAT@B8)1pWV-'
        'vCgKr>-7G_9&VO4b9?R~z{MWoNf^=^FbnIDH{PO{|pVH^^qx3yq7%B?9+lQYsKj-'
        'jDFj+Fz<jL<i6%t9lXHu!x$;ccuSd9J;;y};L'
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
    "columns": {"actual", "sample_weight", "features", "comparison_unit"},
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
    )


def _read_configured_frame(config: PortableReportConfig) -> pd.DataFrame:
    required_columns = list(
        dict.fromkeys(
            [
                config.actual,
                config.sample_weight,
                *([config.comparison_unit] if config.comparison_unit else []),
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
