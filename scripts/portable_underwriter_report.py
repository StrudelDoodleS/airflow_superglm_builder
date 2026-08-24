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

SOURCE_SHA256 = "780ea478389c6f0f1c52b62ca67206e90c1895b6766257d28825b1e832715a6e"
_RUNTIME_PREFIX = "_portable_underwriter_780ea478389c"
# fmt: off
_EMBEDDED_SOURCES = {
    '_portable_underwriter_780ea478389c.reporting._underwriter_styles': (
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
    '_portable_underwriter_780ea478389c.reporting.evidence': (
        'c-qx{Ymehdj^KCy3bj8Rq+OS{s(~FGeC@_nS7+}gre0T7J+s(^At-'
        'G*Q`@%Wk>sk%$<6<MWIV}uNXqQTU<ZS)v?7ARU@#aAlEIkI=dY{lwrKC3)#bTwi$hX=u6AX;EtBqiIvuKVm-'
        'J22ZJV~-'
        'J=+z1k+kKhY5S_aS<YrJcg3kM+b)5ob$2NGGTB#s4^@xNt~?|KRNNQazDuea!1T9eS0>f*c<zhqLz#Ro4(BpyuK%'
        'lqD*tB&#R__Y{UuF(xP!my?tCn}q;A?{aj5=Y?iTRzgqUeum$UhNKAY{^=9uL9{@kD2GS8tS0jsF%rZ4&m0L^CV+'
        'p*|xW&Le)I6(JUx-'
        '71@y3w2Bbb>K1l8@!z&e(^njTG1(imn5MR88|~0b?%@fE|qCQ0Nx*`FM>KQnfGZzPtfCFOm;sagbH|yORZ{I)2|a'
        'U++FafvEx?b)DB05FKGat-gT~m~laGr@N#8YMu0_Q&I2W1N{HAo6Rm1$v&3HqVB70vewY9W(oXnUIV2*m-'
        ')7V)^%C;^M(9=!nrIC`K~;bwE&?@itYA%Jmci=@<X$QI^}*3B-Q2JZE?zB{-'
        'F~=X9E$i+c$@*$%~ttw!8rVP_&s{+N$plP5<Yr-'
        'g#Jd=N9<Nps<CuH%(h%=eK3iAB&SN`V7nsq=NwgR@Js5Ljq!z&?YCX-6Hj`{*1ibw0U=In*LTGd+`-'
        'My>7bhBXH&qMOSqM*LmA~fsW7EWitO#RyVhO4%|H7%w}(1{zv}i<>lM_;-'
        '{Z3UVX~{?Zf4d`TKX5Z$Etm2><%xTT$cV&mVqz`RXEn`5%`bZRv|YJ!hq_E?&RJ@8AAw{-@`%2x?w@c=_tn<-'
        '51}hj)J^rLzwg|L^C^4;MeOI<MZn{`t*Ytai^Syh`rpUlIK~{Et|%^M?VlWBo(97b9+Qz<#sPl$dS~<>&I?w+YRa'
        '>}s&Oo<FS0Yw)QY=>B0AAXNVD!;cpqLexq<YGs4eN@LW@2C0<>(ai>_m5x=n)Jj#2MHV4fw0|JitDiso<$@9I&zE'
        'n1MBV~f_V=>x%03lP@58GzvN8j!(#7jxmd*0l7k{~U{d;sD?d!+Om;d(m-'
        'N#RtugE%*eP{~FA5t))eBJGh?1G)Ox~>jYf2Tj)799x8J#KhZV0LGXWR9CNh))9pHnHN=l{@&5&1U(>i#IRd0y?2'
        'Vi~kOZ`eAWDMdS~O{Q=RBMS`ky?%uzC_bEDoq&K!9jBE-c8^g%vFt9-'
        'cHwiPWS1;ec{NeKT<)_OFKA8Kt+^Y*g=45^>s#;OB!{3C8;&+Lg4iLWU<_`e(<K=%|<UhRp=s}HbKiKH~hj%}`zI'
        'X$h%3m)&2)RGsw^G-+gZ2Gl`5fAGz`QLo<|_Gi`TSuvoBdy-'
        '?xeD(^{4h+W;5{#bQ&DSs|N7>+ONdO_ib@3SD<?=e}tAl;V<zWMsO|;D^OIHpr5pDaVI{Xa5c!0uCy+6)%ERyRNR'
        'P~ay%Y@J<K$yf6%%=pAMyLyId}1Bjoj>t-'
        '7Yp&vn%Ub#_?Qf#KAG3NQnr<XF|!@qEm;<>8Qbu;tsLQ`>+}d(2UrTUiyr5|z>u{=Po9HT2*0bBWpxv{__;{M39Y'
        '+f}kZG(|s3ajnn&1oV^yKNQy`&=tllehR@XekT8oxWa!0Wcc^GX%6tuTUh*~{k$qpg%nah*Tv_eI-mkCJKUNgGVC'
        'Wav?NAb7C?PDiZRrFk~4m3dek(L=!v67fB}zXf7|RVp}OYWZp&$1zJc*vh~e<Q6!Q_FO3hv+uHr3<B}oFvQDh`@)'
        'SVzOt*Pb?)~=XB3s7z_D>Kn<Epa5%3xslzTq(GDWpa%bZ4C?1Ure&f7@3GB&*@YPnV<&;YYl`i?*#s{nL@!6oDlL'
        '~;E1qF0_TRu5&5RAc6sr&9FROHVDXTb?#g{}KJ>X1_IGO(TG^wfN^Rkz^>j}{k%-NNSol~W-'
        '&Ms;4N|4rcB7#GC_XTUB+dnJwC~&M`mChABZN`9=GS<lS|$D|%azkNZF4@6b%zFo+!2NDeD8~6b-'
        '1&?`!6MIK3o86`6y<h>P~Q!uTU^~g_CGGDp$~I!inS}Z3XKpF8Km;O+~Q7-'
        '@Y`P!sk?mI{*XgehNgp3#@#W`5oC8tDMQMrfLPZJU*L&SyLvlRGVh9m0rc#)RY=ku`3UIt2q9X{Zm-aaw8>ef#wE'
        'jGqc~fO6B4jm@LndXWxf7OYE~n)s;Ze9(V|p7ZAhc?xJm*cCzB}0o9vM=^hT=QqhwHSp~^7^9Kbz6Jq5Oq_*^>v$'
        '4_cs#ZvQ&f&D{%|fXd0&!ppZQ{Ff#sn1w?T|x5aO9Zh^Z6@@Yf{uZA(I-'
        'Qsl>yYQ@>RGt)|5@&=q&*Ewx~x0TZF+ps!Uxcs!PgQKjl%wdnRq-SkP-Rdpv;d@B1{=<#Om+@cLp%xF4SC>_sTpI'
        'nzo0Wv|fNt(T!Ik7KyXz*nNodc`O{Sjnr2|KMN$f4@`D^3+$9QO}$fi^8+iG?LbRV(@|L0>#}DZ$zjz_v^&5JnHt'
        '0%sDSxvZl9ur<{HmT)>!Qiwsm+GI<h2%LBA5*6BtwmiE%gvJ&?P>~O*p_mQ{VEE6}^Cwr;S-'
        '7i>&?`988n&EkL<mH@3q&3bx_LHCdj)GbI0t5C*Lqw_#p_E?>TyA_4!~zsrF1aW<WF8jp|jpnK_Dsrn=na;1dnEx'
        '5Rh@EX-'
        'nAQYhvl3A~xDrgw~vDIh?go^)}6N;838GhAM1NX?`r*n=;Jvl7X@;I2)l^a0P+{fU7}MUC9GwG?81u@GygWrLN-a'
        'I*Y;WBeqe3Z2pT{7?#md*6&K#AVI1h4tUBuABt9d7CNjRn{+Zk70?`~O8Ljhz-g>4*A1-'
        'C{B%H7Q1oiHAGNBIx%@-'
        'P1gmReJ39^byM<uCWuCstThkKTo_=)(NHxs`)zqkY(dmSiq=Cg_*WQwYS3S=x<APXbx~_pXfJy3Vv>ZG8-'
        '4eilbB1pnp8$lhXX@yMYgahhNm<)JiJD1hzIGjtDJ>;SoQ~9QL`z8{JGeC&r<Dt+$*&24#0N_o@sohl3O#M2zu8v'
        '$3u#EEjJ748ZFBNbL2y^6Di>!2T*+`Q?a|bi+9}JZ6jzR6J8i_L9YoQVQ*4rdj<TM{Qwq+w?3SWZ$ta=$b5$-'
        'T5S5~$T008oU+}BcL-zB!2?P|FT_Cx)7=6<|D5)cmMqZcjubyv=`c%WB$f*{os=QcjR&0~8Hc70NKHVJYXj(Axyz'
        '25(bCTYVR3f_|9cE$=;I$autTG25q@q_t3~~<qwm+b$929s%k{D?zLGJC}NOTXp-LAJV61nX~1rHHEk`zG&jmRNY'
        'i)=0Kye>nc$`e-#4%=otNNiK#7*Q#vaiTKZ$I4q~0pX?~5xMWd3CTYz1BK~9ag`E#u-xTW4#$;=OfE7oMh-'
        '?G1>~eFh!{L77$YEk*~vM}tZ`2yGQJx_$m3m9AH9#bD6jxUfdNfX^ub60#}~21N<dqZRm=c*16I;sj#-'
        'Q5Ny>Lbjg87>u7ub_gB1Ci)}-1(o1xceRa-d5=5?BJ-Cb%<q&<=}xI!F7KClT|amdypQy?_4*gV8^d*+CO?M+Y#e'
        'OxQ+2m`9zfYRRgZ5kBE6j5xNidLD>0qwC|e6%K)1W2WW*{V)(j`FDq<b!(iqiwnBcrsSaY|uo&0#3Gw4A2--eZvD'
        '0!SODKY)zzlZEuRgQTx7(2?RYE?T)$d8xuU1y)?hh-'
        'B0t(nDgKew3D7}*_GVuZ|bT)@5=du^5AwIZkcP_;#q?3asdpj73U*zo}!DYKA_}y2qC?7FACf$nS-'
        'E2be8|sRCU@x15_SltxT=5Q`sIx^dum66&}OFi6PlA%E=)dn(ftUkzC2Xr8*@4gvDz6sP^PD8~kPif1U2EblsKZN'
        'f>{L7A#s=EYa#G&Xl<XlDg%AnOo{YePUFjl!ql9f5E_6N(2GgVl#m$0sCF@eEB?E7WG}4J!VENT*J(DQ$MRQa0Mn'
        '}&L~1qgmKv3H9~v>4e>2L3J#y=07=-'
        'U6KoVB5v$>BRDIJI2Xxe+bBhpe*`1GiD^MMEw?*rYF!>h+0Y)kR5Q4EjE&pD&O_v@@HIEr=G#Dna79DZw39q~yFE'
        'Lb*VaOVMXsQ**P+BquAQ_{F_ifHLf~eq1`JAn#h$2c3VzqWgwycU?pJ|BjRkL$E^>_BvjO2R)8C-'
        'wS+9Anrgu)R0oP%h@`tt<O<RW6e^ISE4(^X`%<EEzxjudw2@Z2eL*KD&IMvMDPq>4p)KnX!R^j#*uly?r3OCTIis'
        '6|ZugH#BiIp%U;eJR_gCXEk`V4^kwX9NVC-'
        'X{2foXtaxNdpsOt2L`EO;Ffu@T9#L9XwNX*Rbx@A9##b>RnNz(YN`6#*7deI3wSzO*qhSuii8%!X1l#dmEr#VC0>'
        'R=Fj!tjD-55Lqq+UfKr3!Y8Y*!S!C{i<U~+0!Fn!K7|wKBD4>!URlPmn{;T4Q4HOKO{R-'
        'I5zs_wZtW*SnU=8s{s21R=z^rYHCm^a6%1#;t4tEL0D>VD$D;fc0?-p^jMEq@C0=rVw+XM-'
        '{bjbiiuspC1g!T%fRH{1tZXHyF!UJ_dxr=*W!S*lxA*ZIRh(aWNEls<tYH33flcb3)b`~`$raDuBZCevyjv?n5or'
        '8bC!4>}4_?)X~^b|c<Hq{ES(R<zHVP?lzDT%He5C8Hwj?>;6mPI5Q20#g&N%onIEjfzb*PzBoO2cgR)d8ml!<`oz'
        's48T_L4C`Z{;&$7C}-'
        'X~H#o&%Jl)0Q;7TLEm3<>Mn~2weA5dWC;kr53yG~Dl!vs=?#{QZUnk%3)xfXJH=PMvJWs~5Tlt!Kfd;?doCm@~tt'
        '$?2a)DYKtsN6`a#KYQb1%X*vE8!k8@5Xj#qXqgO=Z-'
        '~DP0?EY8^S=6Va>=DFq3+&BBmJ(Sw=%1BicA%3}uG_J1xF$>#mF;Ss>Kaz;~mhYJMKI-VnwoTehY#pBMGj9*@!Tp'
        '*|Yh2dXaWI+U*eFmzlFbie~o^nUnNdCe&f7JH3`mfIYIh)j;jOf858$FF|H#3d~b!!ZIgx|<sqW*iYn-'
        'prWIdZN){0O*j_!bgzrX^Puaspk0s39DR-LcGcl<c~m*vWp$uyoIph6g4_LmR+e%Hp`+z3qp!^UKP|K4M+6}ibMp'
        '4j;UA%FMVX>5=o?;EDYExjz#wiwz-l>F3Z2AVz}9|Z*p-3H_e2;A$3n!0g@UU8u*Rvs~XN~rzK$Zw{w{aysi-'
        'VCR-vWOCb0`9s;4)jfyJ(RZAY4uB5R>4Met9_L^90HKmm1b|blGVpHvsD>VQ+K3t1(8FD#0ckj+XTQy*(q8<ZO(0'
        '|$*58QkZSmfsN&BER$5Uj+6DAr28PX;dA4!W=J5%dEWW(Ax`8u)!b0Q3P=VL%OI>)a7Am4fdK4*KFKg37Kqya=t='
        'IlN#Ep#@_NsSZ0el%grYe6zMKrb-f}R7(ITy=8xbA{hQwv`OFGl>IIA%|cX7x9-'
        'Lz?<%Nq#WYko+?e(!3RT>;8h#9rIwXVv$-'
        '<1eha3gpay*d&A_2kO`lkYvTPv^Sn7)^gl)%Q5pA_U!Ai8^`Bp;&xbz5wIDSLJ%b_fUq5`VEWMH46!e|b@M#0B@b'
        'Si3;n|AdIU=q(`{cZWx5>UWfiL<{>^a)SRB+q-97u`g|Q3N3YhpO`z}d}iH=f$<C(6HV-f-lPNV?b5OldukUU4#z'
        '>4h6QSx>JG)ditBRU2*VliKhMVoyX5}jg^UFl^a~O^cw=Hn6;J|WcCJEA9B!l=Vdstw+uTx%1xzXnEA0nADWuhnH'
        'f&56llp9-s1FWv1+95@;;ku<O)59#nR{_^0O|+nS$}f5pW-X_@?Kp|Pfe3WVsQwMAPlamPv>6j-'
        'W^#k4kp(A7!gRKoP<p(b7TO!jNj^8gv}WgIMz+Qp!TH&;g@-'
        '5S@#HS5hLDH@2(!l$tw;Z<<%MJ($PB~f42};A>pWXB$<Uv1Jb29CA4MNv7^JOs9LQH#dxVau+BlhiDh_*_V32EKm'
        'qI(S|G%blf`MkNWnE(jrP<aco>d9{XsDv;}q|kI5T@uOkTAsm5pMvL|L8^B5+gMW(mY7dsrH~@@u+l8-'
        '3y)3}!9amahfqnqNFn#9Dhq8?aBA$zMB+Jz%&B7_IzZcP2kQi<MdRew4E*BGseDEgfOy9fO=;D~CsC(P>-'
        '{Ed4(ulG?^)1vWRDb%0(|9Ml1wo8K`{tqQ!7T88kdl(i1gg44~nQ&j@JKL@Q;{g=jMk?{6&8Fxt8N|rg?BtOpO5n'
        'TQ-;mVou_1Xe`1)d5CRr*v{Ce@Y|sC)S@b!18}rq#6+B3I8hDCYIA7n_NcO&Ddb@fsvBXl?uy+Vmi6d57b;@-'
        'gAE|6(Lxz+y1jzyu~`mIb~Y@|-W?bYir)Z$kdKA$l@kJg8H-'
        '7!%Ja=cs?~mU(Xz<oLO39=t1vz!oV6UFw>O=?t&y4LZ!l*(h(`8{{66A@0!_K2{SD6QCn_!-'
        'GH;w*wir0!6UhgYwer;rl4t*i{z;ZYdowi#pM(adRm5Q1ob4Do)i>WEO<YK1V>guwW>&vb?3>Rx=I^T61ufQk{OP'
        'HDGKRJBuycu?SX@ra<p3z(g_glk(G~aj^KR*Yr-UqlF(j9do|rTa*WSHHzy2k+U^9xCCO7h%K$GZ~EJgMi~JnPFP'
        'X#wd&+m7?qwbV~buC>p%?;ft9D)R`uC>v-|dr*AQ0Mwb-'
        '0}U{~1JhB#BdZp%||Z3H8oaieY^KjMh+@oxv{$*>CFfWbG=p*HqnOo1RIzfzMCMNUkVBmmF{$3gvYsyxFT#Ux#P5'
        'DQf|6`rbpvOm#X!zm$hK@N&DI@}2(l>iqH3Ff>CS+pYygsXr<M*6Ul$1LM<ixM}|3-'
        'gi6cxqbPMM%^|Ni{c3=CAVxc+h}qZS#c?ZP2C>)0}^;+7t1x1tYx_@#7Md#Muue61_4=Z=A4u<a7-'
        'rQ?d8T*!`5?dk@dwB>W!pTBOmupJZ@hbt{$vbL9s*EnzdqBm$`$*p%tSG{4l%ms%0dA(SGY0cNmKtsY6aTUa%^zh'
        'Tm;TAFmfYv`5P!))C^(o|{L;Jca&JQ#>1E5+iFROuqw2MD&jm&N+#s*9wT3HDcE7xr*XMsPrv3WPa@E5f-'
        'G&Kq56Om^JV8av48Fg{}H<3?Ylxhluy@Tt%Oi)MFwlA4JwJm2Ib4^NRiv2H?G-'
        '5Pgn9f<*_%xWv{tzFTSxf|sNl=hUT96wli>}~P6Ok@}V3Aya%5$kOzxhcWg+6f8z6C9TTu)j6N`go<aBF(g@Rlk^'
        'moBB73mun3|-x%!SBYMyO0Iz;v!2dX}en3<`&8r^(<azbG-_EVyd4B!w5sv+-'
        'Q}Ek5_D4kdh!h=FEG=s*l|hDX2R8mR8pNFiRe{h9YSlttRh&MNzdk;GLTo%W4`~X!0rO6jJVi}xHlftp;gs1A9P4'
        'RF!-'
        'k^!dzgf;6Erw@lCcg{l1!DJWrjjx7r(`ZF3q~u6he}>E<&9~eGS9rm!Y@@$Bq!V8y2+(F$FPVe;Jl5wsSZKi5FDF'
        '@)t`{_riRUgJ!TP(G{loXm2{1*~5a}J;x)A^{A2P<B_=I5xWhKI=<5?u;j@ggG@J>wZ#{xY+_r}Tt<#MMlM$=H^s'
        '|)EPt41p>Y4ez6afn!{QF+nkFIopJ*)$i~=P0^D*9xpogrjgMI>}GBLQMJHd$#3Qz1^6DO+gYQ{9SL3+(V?5${gQ'
        '#Msm5iNn4GzF>;cXMlvkk$Wa%b>Qf$BsUC=GZf3Sw{>M2CPIgcy>ss&6OSz1g54Cg_JR>D2jd~4K0?&vD$ZDZ8HJ'
        'c<D}NE>I&G>RgT!8<&B<_<(@<~U<J{=cA~F31-5Lr#jo0$|G1khaR&>rN?sAOi>?HJ-tOA`SeOR}2Clv^1LgD}!='
        'ZaGi{#EkpDo08WY8J81)~uO-'
        '+tlxwXVBzV^RjJNKOrRm(kcbhAv!Ufr}%P^J@fpm~d4EY1TIfXeu*O{_o{;5P%@N_xM;&Dlcd(%acj9ky<3}^aGa'
        '97^DpNHCXX`d<7-'
        '>XWno2#PMmkW`Nfv=9{mR=Yk(zxYuzA9_!H8<@QjW(&zY03t*W!Yi&^Yq9h}RG!=!XZ%pA5@JpR|t8p<GS{!y?k$'
        '2kz5+1AYSP}Q;k&9X2ff*r@U>RS@!+<&yihnjr-'
        '&eqKt^Xb9D81<@@8LixjQ`mpQ6KNvN6;BTb(Cu`AXy`ko64&bm8_GFHbXMjWS<(%^u!My&bwX{Co~wp=&1i3vf^$'
        'Y@H*mm@_Mn-'
        'M;R!!%bGrV@oR?Q`HhrUA>fIj(U*=pCi*JaBiu9muGBE%PjK5fVX0%5oA^6t%x*)F+63m?4OpdSoY3wTid=QwG@&'
        '|UY_z1aiZU}4A^SMX_wWRT&HysXGvYYn@5J#p;$lvu;2Lnr7*_<mlU|^is~z~1HF`qeP#4U7Mad+LS!^<#99U1$F'
        'MJ%??fn6DgPkMT6gY@8zfa0XK%40S2z?`7NvIv!RR7S(U<v3E&BNtE?-9cA1BCSP!AG;EvbqgKs|C0)nlqOMg-'
        'l~pOA_d@ENVBUL~!K<zPeUu1gDJ9R-'
        '9>lhA@s&9|rD5jG08A9?dFSFp#VRYZwik=ehrV^EBrNKh9IHA+;!ASGm1luLh-QbtlvHbiuGrNq6z?;dB5mTma^}'
        'w;5eZa1yt_;b)foFV+=)7r!%lEu5uO#byiH$htjj{sV``8_fLGifQXoaWzAFIAm;<NZ=~LyVuN_g<jv*AM0!(hwW'
        'ZX!#kF14g1Q!?G56~#;AI)CDL`^9w`JW2%k_Zb`iz9Twxjyx5x%p9KG=K4_R^x_X5^JoMI5VT8-'
        'w)k~KY_cG`C7IPXoC#kq9Au)7Su$$DU*<U05q%7TDjMHU2sy?%+)*zM$SY%T)*s<TqRgINb2vMEeXI*IVJY0ntxi'
        'R{gp&st$L7m^Er`~n)D8ce5p`l69lZjgp|nJapKOG@VY7hx#VBgV6Vn!{AR5(|Q$=(c4Iw7bzlMqI#}izL?A!q_X'
        '=<yw6~Ms{LRye*4e`lrmCqXS_-'
        '({AfoGeDSWCx_OX1w>v=#RDHcXoU+7`CFL^cfjOTUvamQT&X&`6&I`HK5C0ExjL2Cu_Pv%9CqK-'
        '+3M&GBj_(R(9kY1SKxvv3N=NjmqU&~7*-LGA(}2A;B5$0Tfy<2G_`bC-'
        'e9z{$aumoEb$h^p(rmK@ZxU_#Jk5kp#})BGIx%QMp1S$+z8V)9<QsLb93$z|KWqV0NIwfGXP>z?TvY-'
        '3^A?*AvTskcYHEC{Y(@gn9abVgt)AUf%~|vGj3UmZ@w&73n4t8EQ*-yJ(OS7^SCs4-'
        '$v;IYE@SkDPal~sGk}&dqE~C3jq<5!z3gpsYYduMUqNIxjrq0%Ad(wDSp_m(irBdWWLA(P?ZS@A!UKK_<g>hSMY+'
        '((foGP7C{iW#oUOv6-?YCAq{}~P=>*@rFc(XChi%9f+6z(AjlwmIJ(H*-N}}D{<m{+K>aY!7lO4-'
        'hk+}cdfRk2=#UyDc2gxUX^h3V7UV~w|B<<yL}Kb(7MoPp3KY$AZ8ZZzh@hFuY}_K$f1~p{A%R8`F%9V%zal%70jo'
        'Ci&C}9EIbd>|^4ss=Ul?7Pg<aMi3>)eOvdcX3nc2x&xZTNvV8-8ojRp?!_rx9-'
        'JM`Fe(vg>Tl^VhxwG}HhI3A+wAS?V*-'
        'g$3)gr4>oN<b}O@PUoxpJuO;4Cl}pr=k<W7Eh@Y$@Ajob&02X(St!4pwhr79}zv}r+W@y>h@+ifvzDMGZN0=M3Wg'
        'YVtWZtndLLy<#O6x-'
        '2YDQgXuyZSQqlfZm=7j3X3JCci4fjdhl6J{9lmy!_RBkCt=pk>tMm^u%m<QUh6Aj_%f#o!8w*COsza9CdOzw@aC5'
        'P%ew;UXNC<EE}XVhiMEtJ(d0{K@<QR{x4DN?^T%3QGoyVp>j^K`(B~7qt_JgO*>1JC@Y>Ha|G|3dL-'
        '|~yuan@gWL4Rda=eJ&O=LwT{!HsM0O8CvN>8@Lx}hS5gHv}&N%WQlfm%j>U3B(taQ^Y~KQHnhUVgmzI1vp3Lz5Tu'
        'B+%o|b&r7vTDdqMwS7uoB8X&FCa}?sv8&2lG~q;MMaKKwoyxY8!g2z5Tc5+W-'
        '>DDLz;@5Qq(}yqPox}a7YQBT7su-0ZoLG8Ea%9}Z*%v;7M3|FH*{-7U2apC>(N^%DSMILu-'
        '}?5hZ(&nE}3H;WEuLvz0r`xQ^D)VkllLj(&MG<sg!NgmePzA15T=NxTGNP`brF%PDA8<qcG$wtkq=LJNJ~o*K&%i'
        '!O?Ken(**wjv&e;HcOCy@hQqH%IYStS;?c-'
        '$kyQ8ASu6|K$5~PW;olXqn+<t`AX^Ce1+E`s;)z?q%H5t&mfF@QG8Pzj|DBf26hFUyB<pw@l^T4m6nL7wt?O7nB$'
        'qzgb7MimfNB$_swCK3bE`UIyI1E@2|w<UjD0+@>I1Z>_NqY-iNSb4|m$A-#|@q-Gz*v8SI3pRhz9&MS(GI-'
        '>IVSl5dxz5!VKI5|h84=lO*YjNf`!12_|yo#6e4cR#$oc$2?>_t%RLABU&QjETu$`~kb)?Xv2&fM1y^p~m~lrq5$'
        '=RvtzuP@*v=x#yj(ZrD_MnwVUi;=kmfUM+Bcg6|u0tfk+nnU5YH#YvC>;0HkL3pab&=fII|FFK)+#tvx$B#(!o&D'
        'FWheI^C%G=+ZkgyJ)1Eu#>q6xPvZ7Ak!K=Ro7>JT~(CR8~xFIaG$fDWs5nlQVw@2c)7E0{o!gkatnztQavn2F_M}'
        '%vc(%t5%Ctj#A%N890N5Gg&!flmLUrS$%M$9Klkp32=J*wLf6HOektfIZ^M@lBc-'
        '6wPiBh=|9K{Lnn}%o6fE$e=%qf6O!QSs&2|#d@M)|HxZr)wiwu;C{YwJpoj*m($1-'
        '$xROo&KA!|<_bAMaMur+;!x6Kn4cN*eGH8l5N<Z97(HgQMLG?xWq(8PG=?m*jLIt7uq6*~^e;3%CnqIwp|MG{+*O'
        '#9zFUD?MF*N8i=BOJ86>M<r%${KIAjS@70KTC@Svr`l4BN=>c%hdVV&q5n6__miuu62f8og5i7MP&}7!C+M>Vt#N'
        'BwOW@a#QG`YQ|RKCx`u)pT|?(IV~QXNj06Ql;&HI|L}w06-'
        'YhcWH29As{KWxlhZoQ1|*jDw`zgK8b;HEGe73!j81dbY&@?Mx?K`(DG@CVR<Z)SJLoUFK&OE_gUV*&n|`=aMMOo?'
        '4|D3W8fdQFR&)Mg{%hViQMM&zJ;KwAAc)GIDmyTrCy*$SI0tP?l6tO5Np&*)UOgZEz|DCv%w@f+jv@hw9+VtC9V*'
        'TxkPQxv=W7{RDw~SGg$G(WcPHT$q_#m5R^Btf2=A?+fhkdtC9<Q7<%AKtuJ=d}Ay8=wzN9)?PZ;wov*2@Alkf);X'
        'Mt@^Ls<w+-y_5x3nQQm4-'
        'uHYGX<y6T_LssSsnJ{JPWY!FdKVBnZn9338%91(%i&!zWl7c0}@Ep4bBvO8bO@ZQ~=CihT!lMtM*w{Cz@v3ld2*%'
        'MPD}=+XaULmdL%EecK#)58OVEadPf3GtW+LN*+&bJ!h!)naJevHS7BCg@a8n!KBJ=$VzNObX!^uskM-qd&VHI(RU'
        'Mf!f0>47><y|3y=)bW#;6}4>dyyg|9*c$mF%3L?^%1R6@o33vE~d6P;RG_r%1xAok1!HQ+r=llKI=cXBXxHSYP)j'
        'd!5!;=~xYO1NNTOnB_!c02V%tjbw6$*4DFtQ@Pb4V40&KgL{Y%pn6-'
        '(IJay1j)EP)YbHZj}eVWw_5F=xP}994s36>KP-*xH=HyV7*iD79b!O7`Ez-vcLsfNQNU4IV(%?bG<4foM|r_r5yb'
        '>H*yA6GcN{wZ*y9{OFFxj#Cf3rTXx{4?On>X9(sHix8fWXpo0o4tUA~%>meyP5>Rc_2rV2g#5|AGu8n}2I!HdRty'
        'np@flbhUh<naX?H8KrlZnlll6#gB_DC|U1jwKQ%M_X+}7XL2s3iv5yo1d%z%VGZA+&uIket&O&Pfjk3JBKg>61@+'
        '#CzNH)xljK6)0@{xU)*#9WOMIU$H?YKWBzd&nJM0W+E+OadCVOgf(<{_cl*2Zq37ph9`-'
        'A1*mL7!X2v<)@u4WdVl39!%1<(n8Vd{0jt(pk=Q<p%oxp6AJlp!{G@6$Uq?#_8|9%8jTqw@$$_XJm$(Hy!MfT`S-'
        '6B1KdaOQVSs$D_bxGQ$anNtt4$mI?BS&6hFh|7+Z!u&~nJvB~<XRg@nrP#i|3NdLNl$P2kI~(^%)@f2UAeYu**Ih'
        'NU`-Zz9cL51q(Nio<V6kHZ1{o(SF#5eznF*B$km3xlM#{o29IMA<7p+XKFSB&AsFVzp=Hh-'
        'f1w>K5|A}cF&0QjA_cv?C`ow$Bt!5OPt5*5^@J(?'
    ),
    '_portable_underwriter_780ea478389c.reporting._underwriter_movement': (
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
    '_portable_underwriter_780ea478389c.reporting._underwriter_html': (
        'c-rl~+jbjCk|6l5uZWDw$^t3@2&6=%M2S-CmYM3JE-R_BDx1Xv27v$>DG-5<07xR)IH#ZX?7q&-IsFB@A22WT-'
        't)d6QD3rV<~Q?81On8hvuC$2DI(n6+}z#V%-'
        'r0}JdWdc>15m;=F@qSP18~I<NKHUQF)q9Ceb)AqWNhWEvBQixG1uDT0})U%Zqt0j^i6Q#zlS>4F==Id{Lx>L6n_g'
        'S(Hqt`8=6t`Lw)oLw-A*pH0-'
        '?zm)k@v>oP?NjjvOz2tbP0K80Qvut`I>WmlD;XKbLr7E2ii@`7%o~ELFzQnq!@bPph@#+m2+JkbwoTM7W+vjgyzk'
        'NS=^7!5J!P9r|Zrr%>4;pIgEV(SRU(>y<_Ki_Gjs~+yKA$WHzm%<Z)O{G0^J4c#1pfn5@HT~!Pic;_X*o||G^6MZ'
        ')sKqBbe^51QCbd@8GOdc%%&5V<0w5lPDdlanx+w95yqmUUOpaEEs5(%I+rN?0@I0=3(+etXtjH=Fq34Mw&Djj%VC'
        'kt=5Z&AAN=ia>Q@^^iQ_3!oEB?xaB*6*WI4%`(XN{P0~lxr=kw4T_H)EPN={CS^dy<5k|v^2GEZRfNAh<!zd(9|`'
        'Gs*uIgE%%Ym^TcXX$jV$DU4y`3NfR5n=R3i?dm&u<1Z)iNuwZ!z|nTKADthCo0ofQY7=dDEC@%2gkV^x7!Y|7X4='
        '5%nW^lP5$<`MZdp&CvNL{YzlRSTf<<5ho(XrtnjzRc7OY>tVMrT*1BhE-'
        'IKM%f9KhJlEO%Ed3u73TCfaiZVuwJbe`a($A<?oRgMqao#>wz`8?fwUo6r#Cl@mxgJ-Xw{`uweSMLoXgMr)<?_a#'
        '%e?AxhN(3;3w63eR4^?-*g&V%vfBk;{=fOXUCVXercJ$`)&-<?*Kchx+G1WZXxS>Y_GgSahJ@^VJ446Vn^Y-'
        'w@1N<kNB-4|<IGx7$Ax%aPq5T7#zi4=x6lFT!ix=~8_ik(orpZ~l7oTV81yXYyA&UgM-HR`>(fo98l%8k9v`araK'
        ';hXuOD0`dr3tVO0fwgVp*yz^Zi??%gV@r;8-'
        'Xds(wnl_RK!8w{A2W1Y_M{c4zqDK1k`|6lo>Ko*l5u_pGEnY*pjssXGL}fmCn<$7ya?(4K>>YiWvTtT!oV&8+GUD'
        '87#y+?IuN<l)F(pPmaqtifJA$VM1tb;`_liMScNIlEpk1|2fO1z*k%SPP8>H+M><M!a#zxalo#3BUz++0$APCG&?'
        'z+??$)#vrF}TmJn0jjn0#z)rHvsD(Y^vqrUZFJJ`F}YpQQRRV?f-'
        'cGkL$y>y~mch4`{s<Zq&Eyk1lVmCU?Mx%79zJo~13xxt=ty_#?6kU`oyV2GygEf7=ki)st@2l_FwrhHU<;iAcR;u'
        'EQQ()a<NW0NAzbKL!AJ^~Hx`H5HlxAQsb#h?a+idF2lHvr4Ch2%CRxLQ3v&9^Co`qmC$xf!bQGw`(<`qe#46P{^-'
        '1o8ShZec-'
        'K*qKNSpyzrCD7dxXJVRe2sVn;hpm49_f=i>k|FM#=*skkqq*;lca}`kNwD3zZBn~K2;4<pd=NDJi2<W}b(|FbY>p'
        '@CWz*Tbb(hgJwCf^C?nZ_%qlk#53v_!1xDA1qriJDnSoe;%m%GRLe4d}}MjCh+^GR}?PJH}@zSZT!*Z_yBtsF+7O'
        'i`5u`d|~awS#-CqCpQT!43{JyT_Ux2>x=&DG#YU`bsE>$#h=PL-'
        '(Y}7qdpQ5?b7kT||k~JU=;^q}}5MEadbcDYB$HE7Gz|M|<%cl$7{TV|bhwpp<s4c|V?j9-'
        '}%qPKF<FP}31F?w0x3u-U5C6?-'
        'Yf8tb`*z5j!|4{%tVP0L+p@2Q`+wn?4{6ye|iVjPrN(R;gX&E^(k($>s_BACvHr=hW-'
        'S|w5VETa>=MUCYSjfIBC1qWeYv1?J~a#fDC?Bx!diAsiziV%<(oFjsrG#*?-'
        '#z}o>=xX<Bq(X>Rjw%Ehj$P*`onS%lm$EDLXZpt<;_)OOez^ZhP*Xb-'
        'D$g#XGM{84wZG_l+pg9vl2NuO)gBbTx@V)BZ476;F>J=*B<l^L#~4CAtdCVe>XpEINmi4@2HxtmuW7qD1AS@P9T#'
        'vykeu7>BL<1~gHp&w`QjM(;3ON*{qF1NGq_EWIl~0N|5}@FV;y|W#38m_(E9=La#-'
        'XW>{s*A=+YLc!RPvQ<O36(lS~6Sf|OViR%?*e+C6@>W$br%xM*J2bbhJ_t`1s=LN@%cR4D!mBr>8!(F$@tWAji+^'
        'ziR79<>`&ES|$)FiJ%#<VncaG^Y8SpJ`CV2ZZ2%?^Y-|wt~+OEL)}pDGz$-'
        'dbfkeDVj=D8)^wgEq3?PUhYr8ZpyRv8pPZc?li18G7P`|47MIru{%jhkkzNzWJG|~kd)fnzi&ylp=#+6w$0^nYe{'
        '28K)>UgO<)EbCto<{zTnz7Kt)euZKaVR-`!{j7Ws}L-'
        'rt$MSi^oe<%lAWrj}JKM3VgP?hfzm+#A<z#&wK`m&@rjT{am`QwCKFWp}rGksg1@=3R}Ar;}_4l&5WbT{rEcJ1>B'
        'xIm-K{d-'
        'gk~V{F18H%In#adsSHMz`+xYR<MZh6>_nHWo=|ACLX+DbhAF2@ZV+2<1+q3FQpxpm|*?ioNnQnWeD9q2Vk+F$MxG'
        '!$Vs)J%?(gG?}Qqt<ID%Sm?K0qdB9FzY{RPfh^)u#)l26$tFs<d!7OwMu!eOUh7lUJ#!GbeRtOQz|U6A7g0XuQ_}'
        'g?7y&CJ=f|1!I2QKSGkTQis|vO2ZPV>*Zc~o)IAi=fr22(V_m%L@6DZXHH`ikt*)PX=j(f$@q%@f$n;=rZt}-'
        'S2D$^oM%P!sesZMaVY;z`@`122`HsAJcIR)5!h!jmz7%2QDNBxhL+T9ahaQY9O&Rk5kdOMZi&`7yRCJMF%Ran#{F'
        'y~|7IG^Dm4faMh2UgMVZQU|WEe-~>4hs5`%*;-'
        'c+Z9dHOHk?;1!_WMMzqkB%jhkjYJ8T?i)`3MhNXWXx2ZMEZUci-d`lREP;Y2^+CwFDMg2zI5th1Sy1Y0|i*zk|-'
        'BCK3C(h}0iytDA*{qw0%Xz&W#P6<cZe=ID?ItX!;V#+!bc_?3zC?XiWK(Sxa_$Xe{eL)1M_CfJtdr~Q?@-'
        'jJAqUKe%}ju*kDz%YjE;9n#V34c`}{&x^s!g#dS|=uV9zi8G&2_V3dfwWsv5F$EyA1aaNX5Hi>6;A@(;7*1lx)ge'
        '|#ISX-'
        '9O?eP5qf@130)B=9b`1Hx&v)G)M;%a#4e9+UoRXy`mEGt{}66<yX^X%UFM7;X=9VJTkY@z`G?+um4Zjtd&(tc}DO'
        'zde`$#ub*AX2UfqL7#uRgrOuOY|2s5!$sx*3Sx)<mdv8#Oo_a`?KpI#FP+w2Yk*eEcd#_NHfUyTd{Cy-bQ<3I>sw'
        'c_4O-Q3Yx@G!C4~B=@ira)XBU)^)&cr^NNjNP9JV`E`?J$%<5pjYFyfWggtBJ0c!)7L#$;B&Q3Sia@ArWg%^-'
        'P(w+fe9_xfjN;ccO8Ugi_q&Nj7YztU%IfVni~bQ3np3gN1T(}BF5wFgz>L-'
        '#lx=LN7a9KBnXjLCKoDgyqWji6a_o`d|bYv8Z04LzA$B+D{bMRrnC!|JPU1>xM8?#0s39UpcotBkV~P@H1E+D(?_'
        'vcclH>ca-aw-'
        ';k$a!=uyIWt+sOhdCCu3Jg`J@GeL_k@fK6GRgn>Z@u_VS4YPgHu|tKKX9hW@lq{kAOB|`+Lo)If{N+l=E!7R7S?#'
        'NU)XTbbgVhY+r~Q%sU*ebTsj<_1T$Im906U*6NX#CV9_vQaxMUvm~2_cptwe4_mRF**Z?j6pvf(JoKBOK?juS2}7'
        'hxi>`AuvK8q*pX4>Fbd54yJA0+zjSTOm*xj<o(Q<~`>0cWy^5DA)whGo4=lrTI)34Yv{mLyvA|o}&@aCIF?hv{>v'
        'XxhL;X_4t0y~EaKIgWn<NxNyze)9gD?n$%RsoT-gLc973i4W^R0XHWz}C1RIQGsUDh)gmZr_39ar+bCwQfqOqqP%'
        'ORm;_5J+}s@$nk0j)`5L199%@iLQzo(x>#EWd43lv`iDu>HiM()hRctjv~}IM*W-'
        '0_dzOz<ajzvu+w6Gb+8;WVh6%@Ap{^o#jZpRKTpu*7>Z`sc$6q#E?trg#_OJFmX>wxWE&_1e83ro>=(KK3Ybk_jB'
        '5Mh;TFW-'
        ')p$gl8Z3qZfnit8m1V#C*rVh<{HX&1NoiT!7$=Ry_R=r3)<4lKo30iWsu;$J<8xVOTznyA3kJG<D?Wmm@RZ^o@bA'
        '4X3Wx{R&pMaLC9R(^?%_c+9?gDEa1{x2VL0nsP#BCIgJ3??89M_laol5<*I7=q(Ow1)sYFZJvNe!Bubq#^e3-'
        '8PrJiRwR-'
        '#2n{@_1J>ZBAS)lwCNzWHYi*8zq+8koj%IRkf%zhCgffh1{CzkCc_iAZI0S?YOfguIQQ4Bt;u@ywZC3`bL_%4HWG'
        'x-JrD=s{o>8L$A1UfK^SCdFyt^xqN6lN75aCw>HC}X5U;BAZ(l|vTOVJu`v{Rc=pzP13*FQ5-'
        'j{q1!>)@qbzSiVEe-'
        '+I9`xlS>0C3Q=xM9*V4zhbF8ZXR4`w9#ccmng5Y&5C=K+;6^~PUa8rcVJ;19I3R@dx=h1MIl;vKGHw5Cy!LLeU5!'
        '3cST0dA_&}e0$4@X3z2_}TR3uZKqD71iXCimi(pb}$86eL6@cnu4niC6>&U?dFUF$NR^SW$;{B0CWA8aj9(5=;?Q'
        '0%ml4_*(w;;HH4HuJh)-jxhw*?$-E3#3)k84RtT>Zp9DZ=d)-'
        'GtN^%l`9)BX_I;2bY3U;KmX&>(c`^4l9x^Xj*%v{E?|K)RH$^(iz925cMRD%0&wgH{(_f>TARDqTH7SPkxY#_Sq6'
        'mvzR36*}c2LC{)Os_#Nw&x;0Y3($%w*Kfr<0`@(+;c+8vApQ!FWg$dOQ>hD#2&KW4mf9?1|y*=?&e2i!NdjY3+P8'
        'UE%=FPgDmgv<o|miL*V)VaCWBc^m#u_9Ga$<WU?JKqEIm!2ji)=-yAaPPewt@5B!u+{DihdEwUGA8-'
        'G3?{w?V`MuNbwyshMZ0k0{`tHX&zXrf<eHU$Qo%Z@$AviG0=O=8EtJhmdn6!8(G9mEAm)FcFHzNDySrH|yZ04?nh'
        '@@5##zx=}&EP6?*NfPpBJuH|O(dr2tjK|c&(OVl7Q1ND$Lgl}1qQ-x-Vlw2*J+0~q{+2fT7kxPHMwx!Qt_*CHs$g'
        'PL02+izC5^A%W3RfOI*WusFGqcbXZaQ3qXwXNtQ>ZRrS4LRCyc9OC-'
        ')g;EbUCu?sS`w>jf#^2_P=Lmx2gfo*peKKu(QoaqP@w&Y}*V`xgL^S6|X*=&+QVL8kTOqPOF(u>|h(GYW%6wxRJa'
        'R`Tnw1k6f3T<S=1dbMIl$Fs*I$c09u*iHdr{n_JS=xiGZ0}=h2<}gR7RXHKO@gXq9V&scydpYB+Jw-RjgSzrcL#?'
        'MnF;b0UtL~#dh4N3B_&vx`&&*Ge^hvaKyi9O6MLSaW=45YPGODUo3IYR$zc@zkWDk-M`?7NVst?-'
        'dVO5NK%nGiG!t3@{5?&9r)@<b9kU_GrZSo-qUcW2bHEN(F*`0O<xbBVaS%rnlOpC%%{6_G1Jia>&QR52`Tnw?ex3'
        's5j5DAhX__vXfaQdWvbPug1fv*XQJF6am@?dFo#-^5jF9F=X_=i&;YdsCF<TU~ycE-VyeN@!OY8+`1^R(Qvbebir'
        'Q!n;ypZ?3Is#=QY{ZcQJ~d)0xjMd{D{&RQl4Wp*DSNw+7?U*mK*o~WNOWn!G*f3m=fte(FZe?MvNj4+n`}`)SLNw'
        '=oT4=X*UnXwxP0K40B6ZZ&4o;ug*ZUz#2PzuB|i!^=;1r$e@jU@PyxOHdCha!N3#W*PSgrS25zrG*YZH^cCLvcuH'
        'GzepmsnMS+4VBXp5>Wa6yq&L0O*O#4ZE<OIe%k^77c54f<NecdyUdZ33!i?%sz44We2**T#O=`fn?RFmqi^5vEGr'
        '3Mr}ztWmItuafRd*Q}m6`+~!sIXglW%IpK1RUEc<gbfCbR@A;3aYleDibF#}LcC3;A4=V0gpB-zG!MZljOKEY!9g'
        'NThNof||Dn{A)r(%tQA3%g8R;#<WKkv)6tb{(!-9-s@OxCiMCVgyi_bhf`H-wQZ|i=-'
        'qs%I~8KTLXM3kRf4B4ib^@50-XRa-X-'
        'h&ot$svu!Qc?MJC6NOS#QLbA5UGHWj>gRaRLNNe@49l9nG_hbVHTS+Kg*yJG%MH%RUSUgr(-GP$Od+vOakVd3i-'
        'xosIh2eDlPd`2<5=YTG)&UXo8^Qj>J4g$tanj#}!E7@-'
        ')9t>O`*rTAx?!@_lnX!CSGDKCR&8^YNJkFQ@G<BY5@9Eo<ejwM5sJysWd|fZzqjX#{Rfdw2hjvJyql=;3oYhA$wt'
        'UFs0B{1tij%UvGvH4@TW8Wd9PAltzMn`BhoV?4>>|Epj;EW`+9ia$$AY7X2cEq>U48NtaImSQ6T?qO{x?6)Mg?Ms'
        'XGVw+!Ne9H49i+#DSU=v>$f|Wg>@#)$f{Dd=F1zA|)n#x}*%;<9qU|ghho@A3!b6}k4?kpRANDGe|gsDfDO8INkt'
        'R(n{`&bnec=fwl9IzgN$xjz?-'
        'F~ua`!wgysKcm`bj27F$$>}dI9W{Q0OOrlJ@p|pSsTJmP=rdb@b@Qxh&SNz!)Ix^7HEK-'
        ')J$YPlbb!f;HtPGlaeG=VT)nVH&CD}O(6{2wRA-WAgC)wo~|gFHg8$2Gg{mE^V(_{I@ePgEBN?kN@G+N@VM5D<Hl'
        'DuR3|q;hec3sf{keQ*=$Q(%zYUhm&?Oi;}R#SYb&`g0z2$KPRsq$Ix1gYp`>oEF>)BcO_kAhEopEOs@+W@-'
        't;=^qa)LR?oQC&R{DrZd~GiXjfZn#8ge%YdS}t#urVqV<Vgn-'
        'n)Mo%&^01oz*0_ivAX?}zk^Dgx9XL1CwzPZbHlv=Vg2d^fTcRn{hde8{Nl!{F5O*dOr>w+YNnD}h+D-0-'
        'AZ{^Ry0&s>JJ}JCPLO%DXp6;uFaI!8d9*TRx6avA=D_4L5;Dw))<)DI<>J{M-'
        '8cu!MdC5{34m2q?KD=(tadrc^Eb30@NNf5<sFzV5y#pKcahge_yGmS1+A}HeTSWf4@NW2ESDBlfe8`-'
        '=1C=R}IdnJd^YF_=@E)OuQs~R21jnx9K=7NbBYYXbq!K#hV|_br(;yuXVT&<BNT*PU`7U_u@;Rm$E1%dp4IoKph{'
        '%byjqiod{QqYml7)Mh8Fc78*_Df&SyE&p2QVK@+dMzTd!;$2mvUOE(g#A9K>OAYbdG)j<8(@R$=0t+lmhQ7}nQ(&'
        '<PZP5hY>#rgzNdq}bxey;P8Em*$H0g0_o-TBBu?b_!ft}Olr4oE@W^)ueTKaR9C`XO1AWd>~HPtTr5i!v>3*Fo)M'
        'B^vai|80>>=h=jA?J;&NnG5T_EdSWMb!&M)>A8y;i0t5*dh6BWIdI{moJaGET>0i9(FnDbocN0LpBEXFa6{FZb%5'
        'w?%t_q<ppZD<n2BnjAIm_^LFqGHbK2=_KLdx;9eg3~W*;bPIh>)+>vA`(<NU{SwwLY^`ZIF10D+>1kXC(wn=O5Y_'
        'UCDtjFnmhYn5NH7t}D4GCNzG(LT@5gmqf^n4+62y=@5p;**N7Pp@$`sh9h=X=PGDL+kBJGLQ%_b19T4l1)F_yi{l'
        'R_z0`VdYcCgpx)SFnL2p=O>7*SS~&i*+c%n7H~4JUn>E7QC2%8FceV~3BMk;NGk=QTX;H6jyu`+E;-'
        'f92V(>P$S_&Yv+2V&kgv0kjqt+LLVu14bt!npd;)yxMF+_0Q`OxH|@Uz^v3663-'
        '^~GgR|1sx2G2@#Y_(Y4(r3Xn8|1wIDCn4(8Ai35bbHp3s!>7~aw{ys22z?2Kh|TwKakdaSSmo5?=`Ej8+K-'
        'erbwq4{ayrO%Y}<EaD`Elg4T^dZvB8qS>wu<tmUfei1T&GN7Bx!?@(88HB#%#VzD(vA7ewhtRJZ8RT|$+ln1O}`F'
        'hQ$IFJ}}*Od)N+KMFv4jK?ilNeW4Wt(cY{edrXM;J4??ZN*8iSHM{8^tD~W-'
        'Xd{E)5TdzFKt`4Tgnuni>~dsCAxsknA2Veb;1g-(aon+AKwAVUQc<Xfi%}0S@r*4b^KV>kxl03(j3)A3VOBr1Z-'
        '|l9zGBlMD%il8uBan?}MA7#H}IT&VbdYe@(4QlCLszc?n)Vouz9`-'
        'da3MlTnI#P4M}=ep$SyYblFiR{90?L_&QNHBrp!XVMbIe)=*>;xk56wHE=Ly05Jyvd;eFHN+>G(Nn*auaux0pi8M'
        'UK0!4PuQi1gEWePk=0+vB$z^vh5^RdJKw<PQC{`WW#W0x+DbRL=p8O!eTEgfS{m*|B%&rZ<%;)xfo`?mtgffCLA`'
        'Q(?G3pztmc590E(Qz&wNFpdQYEVr;pCee;3`ssRY{Cyie=<w-'
        'D90e)zf6Y>Dyc29W@{@qX2<MpF>l%%1j223hi;O_>}%3)l>QN;RBJA4W*X+gLEjVN|}x<aV`y^^xmD_ja}yUrX-j'
        'FtCHiHj6E{n_oA3)7>=mZ=Y=zo`3c%XK<cvY@R5bH&Re;PL<T#~y;!apGmGKuMwNNlEv9D|xrtpYvZw%#+0`wO_`'
        'ym2q7wDcFPUE5ha?L&o>W4NAt3WkRE){3O&tLvI|c-'
        '8tOXoF`ie5>7arMX^M`}MoBh}C_kSMz<GaBCnu@Qcf@I_g-'
        'kJ{Se1G#7a&2L$auAUiAiaJ5=Jnh6gEx<V-hci0Sr8ol*=n_;y@xU-VT{Lhl+<we5t#)k=|%MH@%zWoUi6ReUcc('
        'm@whe0;i!*Sjwk8-'
        '`GhvjljVz1%fh1FL+jpCp@X$Kz*p}8Tzop8&vtL#ytugNUEJ#B#mUX>e!q|DnylWRzTSWR78@MIzw6(-'
        'b8nQyohbg@XlwiS_L%<MO1>N4`i}m*yS>vNiLb}wJ4t_={_OAU^plkS9FDiQx5b}ZTX$}4--'
        '{1<XV3ro=JmTjzkNP<{MQ%n20uQ3@xzbrq3eF{)-'
        '7J`<?ElGzkL4c{owtN&)>d$ygzvN^zr_4#8Krl?f3e9>}316pWGU;S5Er9Z7R5PduwZFY!%$zCcJmH#yfYdf;+dd'
        'ARX@{h%YO+OD)FZ;~f|iFSvJG6r|rJw^+fJ#A-YqkH$OkA$_+d;Ki%=&)+_N`u@f1SO4n`ZvVxfp6|c-'
        '@%8IxgZ;-(p6?@7={E_L#b-B)-'
        '{e_|8%u#MK{+WXyL9|R0@7F)90NZY4D)ghCGRg%IL)N7NL|Uh#a%sk`uf%PFMePh7HK(yV$jfbBji6F`^nNtev<D'
        '-l-'
        'I6fy$AeHIcB1@3Q#3_k`zx!mn+}rZ&MUyyRc)#X>NOj1dseN7k`SNvv;%X0|iT#a<VL}nQ!{LIZ8Q$I;mj0psksQ'
        '6)?)~$?~NvNH3z7$*iT{kD^?1ON>Je6bk>d%Hm<>75T`14cKKPWMx&XDH~ZVXq-'
        '(m8k2bI5E_a<zS`TPaYf&J!@ev)gMt>IVp^Z$7=OPaUf;{g?+Ip0{B8-'
        '|)9O@61?W^1c3`bfvUv#~ZW9|8NXeGbqsW5Po9FvED7)#qIa)GX8mCTlb(UP>1@iZDy`B-'
        '+T?u~Gj(|M>{;&VdVym;XKzG&<GFw4pj=sLaw{fFC+GrtO-'
        ')EQUs3qFBSHC|B^*6>kuQr(gx&E6>gSbZzPeUhKGfT3+D%{^xPuQ)`F=6zS3IL}^fwhb_yAyPJC`H-'
        'a6ULJseS1A_VGEs(IU7eUQTZXru{N+B(0`^<svd}v-^-'
        'u9Njg25pB~=VJ98eLCgmwgK=X=b5w#?SrwL9aENDI^G~23nt3`i2co6NhqptY+t*PI`zn%ipkLRs^yB+<D0dJ%CG'
        'MS(D;Gj$3+7)<Oj35<T9&wl_Cmp;4C~%3cRvmhtC=%Pd?a$~G@PSgzKZZpLTu-tP1lA5r!NG@g*%2J!5NF}_@h`w'
        '{d*~jXrDcnHhXEq&UYX7x&xPMv+A{5|l_C@s50dSe76tg>FK4o_R$wYXk$V1lX5-'
        'bn)~j~_;XmJ;;N!c}m<)3wq@(N;7{pZqjDJtc)<|WSm6Nohd9$WU)PQ|AO$+kTL6N{wPY_{Q3)J>>I(mAVO-'
        '3zArL9<)T|p#Y5e~ox1_Nj`7{stbu`$iZ?a)jMNd!|Y2Cz|AD>c>DT<bA&IVczx@|WRcF@hyjwP5G|3k?!$ndw5w'
        'sD{yQw^_n-iJz3Yx`~~}+*&Xi{u{Es<BCbEc_)$<_f90uBW4lv=~FV<1c;iiDCiRJ(zzU3`@U80IKNCwS&0-'
        '5yLd4h34N$F&GFeSq`+cfcc}Lfu*FiT8D$UX5BOh8@pOey5BQs4+boT~9GOGw1W+<SnraDId|^hJM;86Cf?Xd*-'
        '?HyO%U5}zVX*0zW4d}IsvPtWoq}Byh`jj<DfjCu3;&hu_K4@gH=G5V;}gA?#8_yV+V~F$loUv0#AwP)DSH-'
        'Ytd^{;$%N_{0$2sM8LbxEW$qe_7pwOoz176E9_%!-Nx-HKTq4CQ>)A9(b{tAJoh{}F1&{pzgvJPUVoB>_F*=bPJJ'
        'hgX((E9EZRuIFOMA&^^c=IG>}Mqq7m#HvjGfRy%OPelcWkEHW}*R%oM^xuEeNUr#+=SmcM&4Fe_HGt$D1ZkvM}ti'
        'V`c=6DL5g(yX0n3e86IP1i^d1OtOw1obrmLkO6to`olrhrQ_ir+MyYmK+V7esr{W7&y(S)zrpn+0nYCD&FDA5k?#'
        '05LsDZioTN$7szpVXkSuIF!d33sUD-in-nI&9DX00xdtqWlj+){~IuRQ9K)df?W>zv*!H@$CT{QZ-rH!;2Q~2S`V'
        'l1qiwpj{V%dYAgBH6jk7-GCBy;jjvmjxgQ>CL9tuQEx;2nKi}DY<ZT5o3q7$>5*Ag1)@9-'
        'H9$hL(eaSit!hGMG4et1d@-'
        '(u(j2fyY1EN^2qOGp3iFf_&LzWkK*oxccMS>UaU1d%|<C(L$OfSviS2t<a;0C5TNm5DlUQ%mh$J8Fk6P@0!+tB-'
        '|op5e1#B}*f~Je!$U6MSx#1tb`@E<ec5`zNrKh3Zrfkc)`3RQAE@*$KTF9qm&ZZbxmxfxtj39Xcogf4<qKB-'
        '*{QB12poq|-N4m5N#{@SHxTzzYV#K8xn-'
        '$)gwVILS&v@niXH}Oj*W^7z%?W4+Mv4};}uu&VW1if@xG0NhbF=bMCzxi>%a~oQoT2m%NFalL!)lC<J^`|%r}ukr'
        'g&55lBjuRl7`-iwAaTgA^r88;ww6V+mt-R>$H4<B%uNI27@!ZlOL1mXp$BsCk@roQFkPWf-'
        'H=V*pbtYB_MnhfFAp7t2_;0ft(5gqwkJphiVdNX)a&|Q=CiGT*qxoyQyjfh|r6n1Hk|oBJ{yhYu3V8Og$*_6IX>C'
        'swBRVl=C-'
        'Y;PNnPZ+kYws6d{RgyMqV25=?!=Wb+49H1q!U+nKcuk305s+tG4D;X_Az*n{9sY*<R79c@EU^h_>0wZP<qbdm6DI'
        'fLVY~rfk!$mpI&!8sNdDIiXQSEXD|F|fUWiKn~-'
        '<GJ>wrQyraic}~OWs)gn2PfW*|5>YlgOJw#jQi&kqN^G_5$ZE8e#8p75)4FV9diRT&|9+^I|Y6&qW5`eOQCb!J!5'
        'Az&Rlv#0FR68V1{i*^XNh{Z4edt*?Fz0G1G74TL2@umCWsphqlbi}JK}!@>_f12kdBm#Yud)QS2wG8Q6>X*Ty;Z3'
        'S8hV=kNlknrH>c7{=~!?JgdE@-Je3JGQ#z}#}cfb^+{v?34a^r&N5wiU=*0g#T+2_SO&&|8uNTQ&uo?F5?f$-'
        '8OfT^s~(=g?kAZ<Ek_knC7J%FgYhx#bW-'
        '?r5ua*!(~TOK#dkGO;&5wpWK&i@3Z)Yy|?sb^s?_af<C+3kYEtYZD)X(GIa4R9}h|H1lCfCuo@NR>S!c&lI6K8UJ'
        'W^XaZO#2wE%XbeMG42ZiGkrMDUxx<-3-M4#sILKJ-Sjk=9tUyWK?SNwC^Vp*($#k;sf{g;1XBGnA1C$($#MX13yq'
        'irT@S+P)q*$1hvLW7n60(8}3rb51Q&8e}Fm5bq<7#v+MpyD-Xt+l(5QfJ!dQ%?=EwWeTIqiwlLRs&5KwIntu8%5c'
        '@2D}Pu^RSCXY4IN2=(rXyUsCNj%tK0a^2cM`N_gS1{m0+fVT?*DMED3WZ%rPgfl6Y>=xR*2;P-'
        'okX+%0T;oO#O!xs$mQ)!zMwR?hrX~5XjYA9f}?~6fuMV?K5w6t)>4?(nHUPHWDPH^SG*A3k{F6ir*JvCl@q&%4e&'
        '#-'
        '>C0FMLt<wLqWq;<y|d3k^B6i1I9iK}2sW1Lq_`AAN=bCtuHJI<5UdR}<8QGW%n6)X)u)7bKXnzZXE|0TI{<1npLE'
        'U?`(kM8lh4maI_*1@Qnwx_o6XGh$6J4Q`?!7Z)^rP@|0zJ%Ag@c2dM5Hm->7%Gq};HvEj8qu&$kr0p(QA>dk-vc^'
        '913+E0o1%2ZA3a|qXaYj(0t7=4%BC<z>13X07{W@8VlR%{A;pWWgMbpSFIQ7muh%006~ra4#lFYqsySWJ`_-'
        'Uj=X5m}$1aIsVAjm$d(McRceFk3x<-?<9@1uwYc)u!?6WC)ak^Powzit+>)%%B+C-'
        '8NQzV5^yn>4`ZC0=agzs?=IkQ$pOAVCwZfQGMRpT-X4Uqx{R$^V-'
        '4X8)<P|KP2DddF{Pnn``f=e>eWr1!L@sHkFGOJ@nrq!~J%y={oRgR_lyesw>ZhHTAu)Ij-Lv*e&ejTuU=O>k*MT5'
        'AbUUG$U`IPJ3YQD#=pQ*XEQoLvM8^o>|C%N3(jryHv3IE*gM3>vU&=>sN#-'
        'FSjlH3P1R^jCYtfbZ<i8qf$`!4CyfcdRZ;V8iqZjmfw*$;jY9E)8#veD8ysGTE*-l;xrS@Uf1X!jCd-'
        'k1^i4Vj>%zIBwG43~fX*{5agA|c&YuGV(xaJzGSG4Wk$Zvc76heJiury(!#JSG0>mHAoP5@##hvE(ZTPOfU$vK-_'
        'L&JMC73(-;LRjwR5X=UV%B!veh=OC*{y@~yGaXFck6m&}4i|*a-'
        '<4q(T!S<q^?Y=!hz@APrJl0;}L2ozu?hfd7SXulDhdChpyLj<FpQFpJx~j%!HcxVpipd0&#e03w1;5*N4r}c0WiL'
        '{fC!#}ftkl1#$3WvB72f*FLpH!erv4#tvzz0K&_OPv`N$FSk*}`^uNC|ux>y}`C}ai}C_6jREJheK%<zTL61U=03'
        'M6!;tePHI9|R%_%pfNsCTNL3h0BJbI$MKhb%pyzj%uXVvPwjdeqIvO7s@(1GZ8Tx9F0?yJnk_3g$87MBlre>b~-'
        'P=Cy!5icL||WBo{w(q0=&hxvU~B3rpb|+F*!+1s~!OMOxZKCH38~+oi67*s%htshM3{aSdsW*PgP{l*{cg(l*<6X'
        'd;<R$k-x1sWc-QPm(!0XzAH&^Xt>gm$W_A<j_wzUg^7V;ISGoO@T&GVdj8KE-i5IQ-ez`Hv!jxm@AO)_Io?`S*3c'
        'E7zId|t!@MWI{2S<=pt71y0=G=0Ud!x7b9SH?u+iel|O9y9cDiXo*P~uMbZZ_NrIk+nRvt)i0}Z2Dq;nHCOusg9B'
        'S&3QiWz#kex#$OO3(7x}{i(4i(ZClaZ9D=2Sf>1v^`IAhxxne7{;~jhEjS`Po0EMeYFUTLiYWmwXwZSJ~QWW)Z<M'
        'K$mVa=GOl6vz(sXb^7(99xU)yDIf4s(a%!Qk87<Z^OVL+yLV%=cJxOeto~gj&OT75m<}L4Xmno;@>}sk(XDVdBZk'
        'a}Qq^sXV00tX+19q{#ak+1peULvd$(ogB@ZRxg8JL2C7ev5=S%AOrkQ|Cff0W1L|bWhYrBn@8R_px=4GH4VKla#B'
        'utk}w@Wq&UYDZ1V>X6{LM+WB2bP;io`&deE}4$Lz7k6OqbS~w@%Hd#y!!g8ywv<&2S9AMR~;x>YPQcyj@rFn;Bza'
        'E&=#gVuFp>@PbBD{g=z}OLuO}Ric^SE9hGY))$S(K;b~sr@#hQ<LgFIB*HdCw9O-'
        'A|4B4npd^6~3uzl3uLR&V!gvPAb)QS{|9!CAPPt78Z9r+vs6X1$QPrdw$QU*CwkZ`>E3aZv1Ic@zh7$Rw3lIwx@@'
        'Es`f5f3XU@o`d=L?Y^e4Hz!n9aBNb<_ZN$E!Jb%?r?qiW{vpkU~Cia{JYzheN|3{F*kxjBjzl8(D!|R&6tgHr9TA'
        '~tos3Imp>_!vah<idZZTFqRj)~R?w$e><28|C)5|rq+VfX$^RJX@xyd#kT#UMtyp0GB(W7?-'
        'wO=M3oFr!B(`P^57YQ{#CG1*t<|Aym)B7nTn)vx;o2_7QgZmzr5bBxQxS&J!F{4=*;H|<bz-ynzUC~o=bF*iu81a'
        'Zy)=rITLWIJcv}<1IhNF%{BydQDwL>4FNv?)tq%H!02&#4)$mr)5qxm^vb40LeM?{<FzuPb!~#G$iuMCI^pbh2yV'
        'YKc$3Go4r)d;~NYf*q*3>Hm9?ey{_#~HE`B#`hTT}4memzRc28V_>f+&n{jmb(F=~T<@>Xv~aV=H^S>_|OU&w|0J'
        '`k>ztXO6AI5JO*@S!lEm6lgxr*O;Rv8#{EmBgVgMG|<Lsh=P8Pc-<2IVw%W28-'
        '6g}9@06;ux~R`^u`Ha?b#xzRu}3<TX#E~@f_)<&`59UMDrq<mU!j48yyun28_45_q3IW`bSgIM%}O4M@%78<@+>V'
        'RLq_$Uf-_XFX3PiqE$RRsW{sHefcMd%^msox8M58m%8(%W5c)JnL@z08SS`A6_Pn9x^{G9bIN6W!lK_Q{^60c^lN'
        'I^H6tI}=(yEiL&`pdU~(}-'
        ';4*ZYQ}n9A;oHR72g)AG<*2Lkz(0~+d+uQ!D^LGTmZiSEn$%z649Zt*b>kz^He1g_vMKj>dEqC(#qDbS^lgr&@|G'
        'Qbc$zQBAvk;<(Uz9b!QnPqM8CVgo`IN-YWRq9;tJzpxu6kN1v-'
        'SuK>j~QLBrmbb<Q`Vg%UjXv@QR&OUb{O1%mUhYB2XJq|Z7N^nn`jAx4oZPh9Z{`GN0FER&9e0Ao<HKLrn?t!<PGB'
        '7%^8^c|-~?0W^vBh6D_)~=VC-$t14#%F`kWw(we%CS8X%rUae4-'
        'WFT`XZBr_*_o1VcLQ&x4&zz{{FB3_Yrv_+UG)KqoRX0jnj^?kF}fzw6vhUbvKd<w;Xn@sK!x)Xs(eZ*zZP+tI~F('
        'Z?E`7H#g#38f77F#<mzH+iT9KpF~00-maT!PYGH&1G=HZPmRsp_L&klw=XK)6_i=eTtQJo-~WJa*5&n-'
        'cW=*nsZ1v#eYLptU)>xlqrdr0<vh?_-iC$Fq>>X6#651E)CbBIuBQ$iSlU-xva<tGE)VN#kI&1t8~sD60cO|5fhJ'
        '&c9C`&Nrz+EycbK2JGdSa?dYo#)wz7tM?r(5ur5uzAilRe}`jwYxSudQ4-'
        'K$=a`xVV8UHsnq*{o`h(wi9@I*8BDLS2Wy;5T+4Kj5DMKuz_aOQHQ>2SU@i6e<n%V0g(e>t38@C8qOs4xSr!2Yhq'
        'K>(U%|#mCRss~nfp?M9<rIbQ;{JI5&X^_9G{5IcZmnV|GK8`I{O62AFpQv(ubnv{kERf1tg8UfGioKfh@WU?!nus'
        '2PsHAW}RchRo?(PnjvOH`TEnKejEc|MkmA#6_+fwOGdmE(-~syoL6C@QP%3Y)f8b;$8aX>-'
        '!iAYg&kIzOF@M3?Ovle50Fl-PACrdKa1&9bsoJb~FJEl=4gaA9K<=E;!K35IHKK_!^Ym2c_)%B$YCM~uRPEI~vi6'
        '%*v>z0y=~!Expm$^tPQXZfN`W&DQ@z+~ARR4NDU7T2K!FisTTPMZP|Vb97SG##BsyRA7UDwc%hB52?D_9FnOve5c'
        '`MemxVNE4Kov-M_iJjwsO3CSYtHwo=L#<}E9Sw)OVF{`%2tf*7|&1xnL@@t!1%ND+?@i+WjJ;{qD^^9xf@u8WRiH'
        'g3Q*)d%TM%tB9NcD9;w3Z{vB!`fkgOTSs+A*G^wZ>nB6*SM9J&SCY4y~cXEGW^oX7*q&+Uobo<|Ud>^EUy`xhaPo'
        'X2p<#i}>t{kK|lw^GcZ9<*v<-'
        'i{0oJ8uI0=AIVv#3SVD|og(tI1J8@ZpQ2PSK6DTiv)s{X$${6cg8O0vy8<C>nH6-'
        '1a0s#tU)a;fYRQ47PIxe|CTB~C$@(<BP+;NlgxVg0Jg!;$aXws>pAEa?$)Z@xdqwLt7i25jYgny5LKa^8cJ>j>se'
        ')gT<!;~gY*SSBSLQ;yb*F!y=WMDvG>0$!n;e9*HAmp)dbGq|>hEHhLYG$K`Zn6WBQ<$dtoE1dCBNOf)oJ1#f4z$?'
        'gV3$pq*bX^v91Z^OCwdT@{wr;i`frl2V32!a~B8Xs+{Fdj+54DdZ|?}H_}60>&T!ys7lkJ5{|c(vwSQ59ol%@kjf'
        '%0^T|2opCd^8OnvKyG3AQFayH3e{$k<FXtJmRwV59*U8*{BuwbiQ32K&%o=-'
        '<Dpn|xMS(cQ2@Q$0Z>xiZR6^x=xU%>Jyl$lKtSp3!<&<>*NYTL+E*H^oZ7?^5u(pT-Hx>$w{qZqz)hyqqIbccBr+'
        'R+;dHSWz8#R-'
        'e?t;llIEWZA8#x_V)Y`q6v_K~L1XFm{_VpnD!{hdmWe)rx5sMvAlZ|p{g9A2hver{^_Ltb`|ddo)}dDHoDxp!RHx'
        '31Dz?oa6H#ys*i_k3eH6KKMOhcFjEA`hh9qjr{)KnZ3Eoy{@2l{Ll`ly!}bvoCULN(#@LBeULZauv@TUDkEDfyZ3'
        '5%Sw0$OIEP73h)XL2D03?Cm|_5q(vQ8S0vAIA0ofpz73pgduyu`;hO{f-'
        'gkG~K=(L1yQb#vhWG9z{eG;C23xBYU~CeTa|uUH{q7YNa14C7Jz&2exhWec3K4PIIm`3;DLtv#@7<~FSscS^dyeN'
        'hUOF%1Iv5hWdO*R+y91L~1)1_Fs5en4Q)~lLTkS@_zgq5gdbh@_2cV1;_)LuW_lN`km0tgZ`Lm)BLH`x{hI~w+M&'
        '0;`+m%moYhujQVP?`Gs3($&-ldbys^W6>Gu+W|re+-'
        '~TuZLZcB^@q<Xnm;MdI#+W8kKNlKWHcntmseG%M`_GJQuj9p~83hYaJ66^+mitD+2|iO#3Pp5}97?8Q!o%{m2IUr'
        'Gy$tVA16odNp2ty^su{6~#_T_Pcq()A^A+OmRX6_HAFf%>_$Eg;^lJ(3W6hQ!RbF|st0g1kDHdo7N+XMI7~w2a>h'
        'F(Tx*SlG5NE6r(|OGHO{iDCN^q6)(+ix{+Iux&5fy*rUS@!`RapYdP__^+6se1bk<A^E7gck+VU-LIV7;HuQdMsm'
        'Pi@iGtmk0(K$!EE(*fXd74SE#}l<WYpf>aTpMW{k9(2{GQz-'
        '8tm$v!SeVbiChRU9M=R;g8+k?eQvJJvc5NN>dp{`3nh#Wpx4!$bXs+=O@XmbYA}9;_X(Icnerj%-'
        '*#H!WWhYy|gIuVz6{JwPvIAj+o8C%H^5JDH528C?czd&0mG@zVtgZb{+d{(6z3Qm&#+m9e{q8O&6stfa<%jL7J{)'
        'wvCQsq73ozogJqdD15^Yz<5fb6<TF-8{M|Qo@ZrtJdtlraT8oHAsiFd^mjOc3-'
        'NkTNBv0c>L5zye@RWNjycK{c3Grj_%(yNC{8c?Boom-'
        'W~p<_{%)o7r?8+^mU_(BXyA8;_jc}$nFg*nYCOv4@v7cJ<;Jqt`T0Gfm{s;DMP&qSgBbNgW+rma+;lLqwwug6a)s'
        'u+-JqWlS{0M&C?CT@+-2++eTwx|jqZ-'
        'p^fue9$c4pxI{UQ1HwJ_t4h3CG22PA~iYc@v3@CcliM(1+3T%@VC5w43{HQD%Uj^fFHw<y*WJ@#Hqd_3qJ7h%{eV'
        'xernq0OI!!$BUmic0CEeQ@q$UputTokF{T}@=R37KhPH`>8G6IFQt?%qYuRW3NgYtAAbJ9u`dsEteE%7z{Z=~Sq+'
        '<9rIOt^|f#w^s}>ZPUT=i6#pr@=c9NKCi7Hp^uZZY=STDb@9E+v|BDqI&eJ!n)>iE8NQ?6-'
        '{*L7@rQRX+BEv}i$4IVCeyNu=PKN`62&d8M906|zP){Kd(71ylIK!Qzr<{<<18J;PC4y3K#O56A?f*Whs5XQQrII'
        'bHWyrj%Zhym)7<fRWQE=#e=fCtn%GLPDwldg1r%{XP^APZ7*gP_yQQlF!}oI-hGmzscf4dJyK6kCiMxjKD0HrwIN'
        '{LIds`OpO15(8u@ywj;S`ra{z2Io4*4WnVym7J*RbMBu4$cfiQ{!{*taw`G2>KC=kz8m;BYGV#}?-'
        'vl&2=9)9+RfU!7R$BH>A6VQjXUFeUBmbkzU%d*5xhJubAh3SkaM7`d^<9$v;)87S1q<+kp29Hk^AJ6v-'
        'ynv<;*>&V7?G*{d1AvL(w>x`%s5*OA>P1>b3Un9zVuuH*qp#W_wgBHHkZC1qJP6)&q?P1lXjnVpMTZ*x`3>-'
        '(0N<+t&>ky}xz)~oz$~Y%21>77bU!nVuE=#G>>e}!v^x6^CI-sPSd$uwl9yW?hh2UMK{!88x!8vm{=T}?;>5e3w4'
        'u2=s-1PP4r|Gm6evirA(hMhgnU?cb+>`f5_@B7_s7KdIcv0VzmCII)VTI>u+;-'
        'hWB%b&WC^#wZtl*`Vb#Kow9b(qB79HsVH|HOD4|YU%OY*7^^jx5pY=W;|eSJke_Rioml$@lt)7Yvud)kXL%%uxkD'
        'kb22Z7_SRFy2i<?`m8#McCGN{rVhXJ9mCViNGWea4yCuH9np<Fwd?z<-'
        ')HXWtGPmHW7^O+>z%O!Hg2jv6Mg?x6&_?mMne`(?A(;((yR8Gk1Q0Y$HcsUuhyEKO{t@D^Q6tO`k{3`;Q+Jx~i77'
        'CR+8fCNw?VZbJ`vHx}`-X=vXy<5;g3##Os=%3-x8qLt{`<P^y-'
        '>e7Y=0^volpV>IyvcJ*gdfR(JkmdBStKJDzRcystTW>py3?N7`&E}lt{;<P7o8|G4w;Sh4(du@O(L%-'
        '@ru)6ScRy(qvzalPrMsI2HRCLe(`3FV(vL*yTeBeFMt3(uj}fC@HisbpvqkJ)YX+@o1t)xPYa^~4zr#61huV2Qhh'
        'wN=#Bq}x#5Im+q*alI^(w!B<;v&mm1V=TvA}+Y@bP2<+TAZ@S3GY(|HMXz#h-'
        'p&bo?scwfKvC+^tWJ?B@HA>b14Ltqy9XRcNOKs-'
        'v$ZUBG55d!SdL+?VU8r)F+an3!_@c$%Hjf%E$UWT(`S1M}Oi=nE~>?=g@g@CJiQBB-'
        'Cg4lD(bC%(cU-{wrF>EtkOJJwdxwB=jDP`4vr)%=ls(5=;4^Jo;}q}N)Rf&;Fp=7D2T3C=Ch)N+KUdkcLknCB-al'
        'T@Dx%&XUJuTj7gOjU+<(sNQb_uB&^GStc7DRS&}x57`s)=j^4&N15n*MU;h;U9J56cw_Onj`Y)gCX{)pzQ=y^L0A'
        '~czLl@yZ#-8^5Uax9kSUS{qE~458mp?^3O9=u7`M%&LxL?vV1XW)nKGg2o_Pm4;(kPTlXNWqDn!M^mKLGb7-'
        'YKMwB$khqTY|A=gvs;OQ#oxS6s9o$3Wrpu2F4tdY=+>kcdzObMoKP?TRf5r2p_e42ZGo)$<B_H&`?y%c(<$bH1se'
        'kP0R+&4{UMScuR(jA`C6Q*@dWz_-NWu5$qqU29#W2k8a1WH$lPZJE~RkTH<9-*ijxL%c$>_a-qPV;=UFC!)PU;OF'
        '${)-=9zkW8@fBfY6{<{Oy_aWJr-e08YC`)5LM(i{^!Xxtx7oM`64p-'
        '3#b|8vGJhjaBbo_)fr=rUN^@B=@iK)QF0R_3D^RKU7h2UX}!J-'
        '7Zp&1({lT>n^_KNgv4qm(^UfF?wOm*z%$3sQpv2!$x3+IX09*Cf_5_rL)^!xe&w_$?Y$$gsbM3oE{^P%xFe~)X9F'
        '{Ve(D>$%co~d~kGD^WKvSE4HD*=sGOQ>yR5uhE9zoAYt9q2{tiemd~n>~$aI{X1PVX!mlYAp=`^3E4kP;qYuOl~v'
        '){VZ#@EASF1IJE&a!q)}DaPuFSfi=kG{_jfs;&z8UF?c0!aN>8#*=&-'
        '^%zo@G^&s`4_BXhY$|^d@i@=!O$a`Z?6Xr_hb~#VGqM;y5X2sR0yd(lMI7mwNiH~*63uP!W>_4EC0DR;q6!0Wg*K'
        '}l&(IYi>{?(=n42D9g$A^gwD!l$9fs+g(3E=W1rDuNBl14~WHbO*Z7Bcw7B?AmV2&00v*P?-!%xYb-'
        '6M#+1OKASp!lHdzlE8o#SvF&95~r7x1bzf~fILp4<m9ACPe^KNFM-3hO}*k_FyvB#T;-EqHqm}am-Ob|hz=|k-'
        'B`nu_4u0c`tY4ofim7}M|nFr*PXTxI=uvuJfIGy6OPMsEQnt1?_Uq@tV|c9d{AsNkZ<T)^!DHX`+AtS129hnX(@1'
        '$(yjr>Aw353l}E_e056CNfD2zanqAh%WBR%t^iCaU^9rg9{*CoM>gHYK$0hJv=whC$UT7~V_^;s*{-'
        ')8ys14GIxOi!3x%ZB5f;WP^;x|47*qq{ZS`~D{L!6xQ1N?rdn6I|HW|N6Hgy6O{wYy?}w&i_+n`T`{>7aqnQYjvq'
        'NEREQxb<L9+iMv2oti_TWbnFQnaVu6Nq=>d0?Vk_BDv7dLb|p|!BqF6;IPG;VinH;yAw-'
        'o4n=bRz!Phy`AC>9F>3u$Pkte8ZfZMNP@S-'
        'n$J)OXe=eqa9%6nX3ta~}Gc979tgE563&K%JY&3nPgb60RhLEO2Bark&$PyZfE%{^>5S+{gNHPD?e7kE)Le}N?0%'
        '%FMvr|tDvfCK0_`q6?vlD1H!t1N<2T!GiuY(V(qi~<%w&esgMUa=?K#llUgV1%ua$hWPbyv@yeczxHaoqwM+h8=t'
        'Gm76&3|q($j=BCk^e8}!b++~a$*$R`S#xJuoJAA0X-!$(ck;K#WUbqYb_f(-1+6HsZ<`1-'
        'qb9JK<Vpf9!y1UU+H79s^QfN9Bg}yU7LQZ(Wxy2#zSh1^|FmjF%G-'
        '1T3fnno?x}AR9ILGo5Gw_1SRpHPm1S40H&BBDS|cx1m9WHMVAn_tR`U}`47`0*CcKwYG{l2JGMNkpF|Rbt-ZDdyg'
        'GUA^ocK0H&?QPZM#b_a0D6~h15u?orQ4mJ_prE1C~=Oro(HlSNvtd8EA0;7?yOcaqHa+#Bd7$i$L0`N6elzIh&<`'
        'QYT=2`aaNQl^q@BBFUZ=wN;td9Qw!+}+dS%7UvZ?q^OBs14kfi7sEuvL3|mdV)IN|5*YeQDh*;H5cB_J9<5vhIh0'
        'k-'
        'O6ij`UPmxA>dxEJu{$5n&{c|6r93wMgzNIN(dTR9}FZ}PDK;*0hj$aNzdQ&~xDwjYUfF>O6%V>U@M#9t+mBNgGxA'
        '78OPAgXS;!tpfO2P7&ZEjj5+r0G6H_>DCsO)7W{o9f=HOvMgnOR+p1$+3?j>dY5_8Yz#0Sd##Qmx!AI?76{gB$p('
        'rfmXO1qzJ0D&Ww#y}cPc#U{+n1C7&<&?dc2CkYU9?r8sN#~g}`3@woZrQ(eTQHa^yRVz$2BRul?NAZPjA<aiKFw2'
        '=%gpus3wR*%_sFkPmHj|0ni~?M$nu%PLfo3xJ$xb_`Qe8j;sz8&_4nLM@2R97K?$4%KMB%GjDw|qa_(F9sy<tr1i'
        '<J=hs@LmLxh^*IFTRUkD=XtMmALV2$bojFb+P%h+ye4txGP#_@j)ED-Pkx>1Xor3>(7pEppO`AGCLt=%=xvrwg43'
        'OC2v)XdMQ6xl@0f{+-+qrj8oZR=p8;$oybhS<;l9s{8)pxdiI!c1-'
        'x^cg@q9uRWMReHeC?X8m;Jv>7{MVs0<pZN%<s+$vFkAFuO9Ut`VLE)|~XX@LzY*ODXxXl#zP5)KQ~i?BO$x&>hNr'
        'QE>BaK(IWDlu8hh@tRkWw;tt@K|)TO0PI;)83H0S);l}O_GVTdDRcl<pB*Q|p$qhLlAfb=oRVQK3J}M__bZ5ob<X'
        '3f`p8>WZnjlL@ad#TK|tV(-'
        '0;UJKdb2IDV}~uxF>*tf($t)3!Us(^I<rb;fWei|53cKY<jg8=b^Y#Dq+qh@;F1!HAne$LcteYV|ZlK22+Sctkit'
        '%5!Gh1`n|-'
        'K!)+h+mbgyxtF)Gj_ZS<z#;6sKWSqVOA%xnJ=|l2V1&%qA;>D54dOBD@2W7MVtS8~^&6rKg8F>x3+wnl3v-'
        'Lp?4xI<_^K-'
        'Gh=pqo_iByf|&aJ~%getw1c6+hp;<ly}#cj)*{vdv*wy(eIkn;$%h`{S9+X1U173dbrw6>QY_ULFaOj|9OMD+qZ<'
        '%e6GMc+n}@8gd3;J<_Qg=)0Gu_YT&y=1Geh@BGI^Vcj{9ZYBcUsb7aG`313&>!a+9M~g6A`y|uk+Rmf=s{DVBPve'
        'Pak!1mt4dveAnFTNrJ-'
        '}%gthAQGJTb3qUY;t#T(Y}%p5A4IC`DDR0i!D+w5Ay#6j(8SRQY^EsPl^gcU;^7we9!`1qW#ZtEwJt?F${hqH<-'
        'J<T&D)EHA7!@I33CFn@%8r<z1FcJy3nm&s;<9CkeqLC1(R?VQ=b1jPjH(S|Ou;60W*ugk0wZm{4SkrX+$g9e6QI#'
        'G@>;hfclHw$tuNhSZI&|_|=gY))gJo;_iv3*QF`t}PrN1D^A@L&D8@7L2@b%0!7xE2mM_UUE#6X)k>@Yfgz1YAWA'
        '~RgQt+jwWc1#{@5C$wL8`82-'
        '!iN%=I=a<xI|dK%EzlopFiO!m7X@W_1aCyW$kYoV&+i~Z$ZL$uH6}=$Vw|8$0x3LdU<C|wdcy2MVf2&XaB;Snki!'
        'l7wM_;>h=vF8*wiL3xQR3wg@*FnFVVvgk<$PqsQ}1qXLzeZF>&DmV0_|$tzsDQcRxOU<E-'
        '^mm`~iPZh(E6onWkLEzIk!28#A^KFRWda+s*1CRGuw*CTqO24_<>x&~=YP3s69hq!4^qqWJXTox_!Zd#n+0C@#y8'
        'tGYrQIdzYsZ|a31)F0<Z|*v&aWFpwzBH*+Vl{%RwLI-7ICYc`GovU-'
        '_|l`G{SP|To7{r2x)%U#3yA^L)`8In)hGN@HW=I*#a5b-'
        '5OAiW4NgI_fyU1}QlHks0c|%0C<6wM!yzM2V<7Th4nQ6{;zMV}55#Fz1FxBL&~}SVguQj-'
        '+rMZ@Qh!|jDP6)(`Y%aP_{)_vWW^(ihsi@QQHZfqIDvWx|M>3pt6oV;Z#G`G4&eI%_8~6Y4zQ!cLvg`<p!$OkH=K'
        'lDa-8ZRd+dSYXD``@OG;HDaKlTUmfJT*WF5BK7<7-'
        'OvG#hQ`%ci%dkUJR#p=*e*_{#eFQ=tMbHO0F1dJ}794LN`rCsA!ii&dAiP|WJeMu2b7=|6iXd#Sg##}F5=eng<hp'
        '<~(S*C=)svyt=^(xdlDS4DgEG5!PS>;=YJh>E=dxk)&@oO;s6yF$Rkl-61_c4zaQIsFZ;-'
        'grFny_CzOEf7PyU~kR@1MVY{Pg{c*RKZee|-M-<>URqyQh!$pL-'
        '2p=VC0q6;tmItrN%<W2H^96KYt~?d0A<E{4fFe#LYHr}>3U%Ys)zgwPqL5fGndMgEIWCw>*5R;!hrVCdv@PPsa-l'
        'C$HC6hKU9(1G*qcnkn3#BCe*;Ns*|U`jD>ZZN%^H9IQ4BEyVHz{|4!h~+Un+FPCt7w2;=4V!=Jf+kN(0hzH-'
        'S>!QBfVFi}DfVvfm_#s?SK`*4&h1;B+uwEiy<2zNUQ!sJ9_li&=`^D^+~u%0kt-'
        '_wD?QrC(ooj+19?wJMKynt%>oPxnhnm9vgE0T5Bj|wdY}0_%58G5e`w-W$qK@+!Rqr8`?R!+m{#niat;TC-AML9`'
        '9$R3GNFfDDY4utduH8YyV<+k`>pYKJR0wiP1dCDQTY?Cy4$_&2(y%!{E51Fn=&Zo^Qr`jes<09zO78+w;j51TiKF'
        'VtixJ?`J>n)OkldAlHw-!tkh#n#<{jxtlf4a+zJ_O+z^Iv{da*h5VtIHpo1?7j7h~*NoB&qS8lK;%cm)McX%3-'
        '=t5QtSfoQ5lMIQ~gOLi8uO`V$(nUGX&(JRp{TfBY1$Kac?IXdP@C^8jf9S7XDq(F)w+0qeuQVVzXY}`>pCn!beH0'
        'zYKRp|KM^%zH0ch?z5b<vKUn6wdpFsKXRSC&m49BPUCMYa<R}sXo9X8RwIT1!oA)f|?hM!C{sO8mZcullBsPWXXn'
        '`5b&H0Wu!*H@aN?a-;tt7Y#zpDbk7Qr2|qa7_kNm$|LhAB~>k8OqZ2m7S4Dx$MZxHkYO8ts%FO-'
        't=qty77ALkt9<4di`q#HfA*f^eXHl%@`e%2u^gBhaSlz@h5K;4z!Z7bi`;bSq@UckuKeE*RQzmujBmZWj1g4@Qpd'
        '0xd&4m)1`O!u&k&luBdE$<;5!~Qq%%pL|q2L(u@lg`zXAo+MhLBd1xF0nNN|s+)LWs38)jFEO&WmEMg8nYbNvCO)'
        '{A`pNWQ<%zcqdo1A2>*n6H=lqK-hT#3sHC1M|WGFI6^w5-'
        '+F?y}&p@%tdv>nHM(ORgkz^pMny>eF*?_ipPfxK$ePdVyAJw)pyrkW|^S>+}XA>6Twy(rGk5B4>AP;<(N9{U=FLq'
        'w9Zm!RQM!CtzLVES5Fa*jagEWU5bIjV!OR%Y{57YxQua@zo~VzPcx)y?(?#uS&G-KYk#$J3e=y_V)U%!cx-iM$C7'
        'yTGED@gnB8!WW7D!*&6TM4Lpx?=l0gt&e(b$r&|9ty6$gw+t(<#w?c_SKZ#0GmoDE*mS3A7UuNsAN#Q5DW(uOFgS'
        'H1qX2P$5=4RJ7BW?#gk{PKu`Rz?k`-(jFs|r)e0nOX`<Ho*7a#Ve*aD78Q`qplq-TN+O_r52xd*9in?B3tqsZ8Om'
        '#Oze15p>Rj!iDtc|Ep=eWqQ<2l6oK0mi8&LL9%b1AbHcQ$o)`O<oaC5cl+&l72-'
        'N;FKv)6dn`feE+s)kCU#p8U7v_<yU(Ins7(nJ<37Wq@|)XK>1uvKt4Sz%^(yd|TOI7J>%#su2)m)ZXD#$0?8YJiu'
        ';~(k<KojD#8Un7t4M}z-w`F`u_`GSnME+5NHREF5tsMTDqU7S^uQ$to63d?nNZ5OlS1r-'
        'IyAkEcaC$9+$D9&J#w;4>n2hDcF5C&uU?%;0}c%rH=M6Nm}lKfV7>Mw3R<6a4{_9D30itn*9m`KqX2<hPYv`}bq{'
        'WQs#(~~|HI*1MBO6tg?wxEu|iXa!?c#Nff`N35_-'
        'G)M`Nb+LP^7p>mFdmK<igcFcwz6TXKIxfb{0r0lPB~PR5~eu(ju}?M;(;o#%QF?9j71TOtAI)<-'
        ';)Lpjb<l4>dFe{s!~wa3>g5`ZuT490JzRo6<~dZRI}%mF5wUBa0TRss&L*)Stl%P59Ln1i}49@wmzc3L*a2>jc5K'
        'UjDrM+si{B_%q0rsSM23_%SZ4#q%kZh~my7pAS`O&oX>SeUnTL$(m=CvflS_3+Va?w!pXBpTme=-'
        'C#?g<v^`o%`YEgKAGR_dnYs(DL02!-_xuvj<-V-aR-F$qWu-+Ks=-'
        'hY|e2e1hH0okiWZ5$spN*F1&5541o^A_LAQH*Yg+_7j~t)~Z;Z3(RhXE2@|C&*l4=55#=x5)vkkR^YEOTeAsk>;o'
        '8DM(&O3!9dGJ7UXF@R%uP7x_b=LRC|k#v&(e!%E_N{zZS%C4qGyr8W^GsChzA|1Vz5eW5-'
        'ZznxiQZwM6rUKRQGKO_2`P<Tq+-spEwTsN-'
        '9jB1hD8+jIQs3@IqI(BT2OW58cZ(o$O~yrms{w#zNk@}mtOSwHDY#yd(!@QJil7>LIYK4QeeHjQO(g!ZYvOJ!80Y'
        '5br6W{3W<z(0X7{USi`q@L&o7|PH>1_Cz$rPk1$s$FO0kR8Z;qIUPF1q^9MZ<-'
        'i6l}<s;w$nyO$sxp7Ii?CGDnJKx<v<T?eYTkEkU71D#X^aBR^g=bvK&8>X%q{^)z$(2Er#3)??tL0sFz(wSka)2N'
        'C|XZ?4EqurAR1o!X9JK6e1|oGZT;YG@*;6n8LJ`j{=&Xgez-ch*Sl4=bTJck9iE4)s>z*`2nX-'
        'm+|EJy1N<gOiq)CPy|sTo3^>kkaJ-{Q(d$xk4+Baw^gbNZ$~1(!6-{krg@2`_<-'
        'LdX|3Yos>aRCzn^K}ukpL^I(Ganuv5%@-D&_gnV@kSo$(K1p<Q>plvwc)15=K(iAXFF1Ly%>VizKry@-'
        '}j8mS#lF_|PY+;~+<eMIgb#6XQ<h}RiYd7mTma>#MXXi@0?^~F@YO(JRs#zpzdD9wHi@zPb-'
        'a4wRa@%`FmlWS2kPnc%9&^weC)r`Qp>ZWXCqO9Y`3yFdt1GHX(cJq-'
        '{3UZJsK@o22Ev}NlT{)e6X5p=6CLy@F)*dlKFq>`uBTQI;Jml-EU@t3nWUQ#-'
        'C&=y{xh|}0wF`pyT0D>^wa*V2{sMaj^SHUXL^k|DoZ8U>3C*r3b)7=x>w8s=WN{qOJ;Z2hd|hqE5!hg{I&!b!S9*'
        'hsA(%vM1`R&xeJg95SBF!GdVA14Zv^hJ4?`sQ6+h48!v_NA^2rjjnh>9a^$PHAP(G8zg!C)rr%oSOtiAF(knb3)>'
        '6|EkST-'
        '<yDwi`RUnS^cj$!5*R*Nf~X<I~FYAF?pCS8F>dY5<3`f69a`^?|rs72yLAfdq^X{pA>6<<%mmAAz1Oz5tA;*mn-r'
        'R4b&OhKb9ZITim$Q9nl6Zy)3<|oPkpm_@E$a2@oeH8mC1&;Ge4keqSrU+0}5W#6W`~Y=KQWfE2!{9(^kE+r9>hU7'
        '&xh`2)wI-'
        'T9Y+hjS*>nOJJ(pyushDW~oMvjro7?#SX#OOt#y&e{Qpy=+YQ93dX@!f4l@G~)t=iwnv2elL4rj3*R%z*&MmJ%F9'
        'xIwYi061*hI#D7AaR+zC}ZZg<3s9Oc_H9P;kAlH$|9@j3RCsKHy4eSY<*Na&_9S*dj!iDn{zK#yFEVqL%WwxDbGM'
        '<){Xzw`I%3>tIcN=w9z$_x#5ru)`01!z;K#R(@%?HEn+I@!c4wL6h1qvu*{&Z><imyCN)H1^DWt1PHx<fa>cuH!|'
        'c?7Lu9(bFVVL=$B+1CO3@iZyBEOcThAhWdu4nrgh3vYVd)?a5hIO&M6h%RpSr`}PcSaKqYm+y%RurZcP_=?LY$69'
        'Pg_qoK|C~^Rf;8iHB?9!tg%u#{SNRKMdg#xoS76vU+M)-'
        '(Seudrp@x&L;|$AiCnAtmYZa<XU*i37p0TJXT#j{OcAb0B^++>Q5l8(5z7lJPeGUIX&kpm*o0LLu*|KGi&9lzTn%'
        'W1-F>f;;JaCl!WG+0Acku&si-pv1=2^eIbnH@OzNUAa3ln4;6Oaz<GajJiy3Q))u|_FSASdd;eUiDnfvfiqvrv_s'
        'D$z#OY6%!wy!*+1C;9r1Q@+ba+fa&JnD&rr%wEe7@Qlp;^BwYo-'
        'E18OPpLz&6J<O*#rVQVj62J(!RH6gWYI&wkY6CXjDi?^6tb#lHNBC9~a#7h6Lg8C%X&eQckBXuTm7~iWjqaJQ4E~'
        'zh#VS2cbo#0p|tv$Ux_$q|C+qkj-sa4gF~SEX+Jpl!xjtb566Cf?8c&EsmwPqUTFxM&>8tj;G~hW~O1=8RaZ?>t<'
        '!9`t{YDC1e&Iw>JjWI2ZGEE`MAa*JhZ;#v}0rK-HkuC!qCn&+_)0W%z4&kLlcCx;tg9H$im$Yu$BEnsAA{Cey8bS'
        '0*(sePw1+qE_CPRU~^gDLN%gf4Lms;%?0bXI=5@%vr-C1r3H&0XyJ~DU5t6)>@e(jfvaTbI--|FI`seDovYd-yyf'
        'X8x^sACS5Jw;Cb13{hLsp6&El#Pln4bo#5GvZFEyu`}Rrjn%JQkD<Ny$y)ipH1(ISS$1TN18>a>4R{LnZWG+G0YT'
        'LH<QIR&&h}Tf(T<JJbN=GHTlj4twb3J5RFzL@XT%XyXD$sU3dQC~do!3_lq;hJMII)UcHML&yb<e}{FQ(jXVyIJc'
        '6j;XSz0eb@)yaBAKHP>8S<7@22+?ERu6=|?HeqaF;~`s0Xn0^9Zz?{aURIO2(^C{SNZt8K#-;-'
        'ODDkL&y1UMMGf|DYr>E;H*=4(Fo>4(oYxzROxw6(#M?E1f4K3XL;dqh{KPbOgoo@0K31Snbh`+tiQIU^&Pk<0c68'
        'p-DD)dM|{B}HFk6z``*H<PErbwnANW|hF-LveZ5I61-$vl-UQWyy1m2`|Lp-wqoQhLZUI9Z%6&V=RJjv$mp18g)9'
        '_GvdyHNOq5a<w%%5`47fBwr4s^={XEp`B3rN$x6vEIlC0AP`K<vt$A6m+Wtzwl5fE+pM8QJL<w~NC|s5i}>{bzx9'
        '?;H`+QBe_zty(2=~Ar;qZXR`K>7qb3u~Sq+Eluv8)vGt!@aQXJNu^+WXC@nGq<Gz_izu)Q+&ekYNu(Y|T}g~Qq2b'
        'PgR81Z@6fFVgTVaBI@)ynOxB^Ow(Gy|2ot^XnPLfAsl><Pw_kJnx(Wh-'
        't~yuUuCiJzANx=V)z|FNOjWz#}id>(=i-'
        'CZCT?N<o(kUNSgzr`N>r4g)m8VSBV?0~hI8a*|K!$sL_a|57sD4(vJS&C<x}D;zp(hl1ec>r(u=j7{i9atvd$4n-'
        'I%a*2^x6eqHzzc6s*@<3yWTT8@`ES--'
        'v(Ijqqwk?cURq-V74?Bdc9RK+%-z7^wbAgxZmz@ayyF9FWxI*!HHp9?HL-OmxQH<9YWF4yinb$%hE|x?8z!i}wij'
        '`15KJ^NO;}SLte5Tx&7=tE)P<QUGi+uKF9Bh0`y;((n|KI-'
        '$NNiilSw&bs(rUnwxMXhBl4T=5lM2@t46RO+G#zjMM}}6bXkY(fstp!JJy9LO#D0j5vx*fkbA4n5Dcb2<#oh_%dN'
        'OQ?-qMLD`AKZY=ZZp^y&=$?@*Gq|gPQg-'
        't%BeFvZ^JQHhweXtFMACS<{aTzqAZR%WJKpyBn$&vU=CpE4lrjVI`Np?Uk%gb|At)ccTMa{v7fePU5V<gEarS1IK'
        '_aN9uI^KF=rf%sL%6ITqh{9L0W<QD&4P=F<n79jUqVqRcQ|7j+7nc3CDT>FTI?*x9qJ)Ne>I{|C~M(GZ5g)F?#@7'
        'ER4xAATj@(nZxyX!~TIb(C$ipUbOJ{iY+Yp@g`qL%X-'
        'p{kJfGaAX4Z4>FI9*htm&&h^9oUggR=M0IY(`uRPI%^m2nww1o%*{m6}@qcvOGKMPyL7(MCs)jtz<<7L;g%Jyj&y'
        'PA&;d%kMN&dw1s(Xsy_87@o26pS^GHc24ETP_#m4{X$Mm{N}e3<p07kCJrVE(@T7Mx@)Ykl{{l&-'
        'Ir?h#`lLz0@|D9_cO0>^vzTpX)VEY%ac)%`jwiB7YocjfF6z=$s=Zdl;unJ{wj$~|diQm>Vo#H-'
        '>j<!mR@dFy$wVTeSghwbR$o+OQLzFAZA!CvIPy)0OR#8dfPJ&3m3*Tmf?Xh(X`jjHwKa#!_5p!KN3gRt8!o3xdzG'
        'z>v1Hta27yL_69r$gPtpvW$kCXTLS<-YNQi1vxSwW_#YM^a}y`~+`^H>)PW_jixpxO`^`2N!>DqT-'
        '3!xWpAw1Uvrk|N38HPl<g@nk*KNF<MuAT3JaX)tkOmhAd_-<O!41JG_5Unz??+n8SXF3S}5+_2P6;<Gp-'
        '|7QApn_2zW2;qv<;Kl>+8Z~Zkt%|_|I)IV(<*SEI7kdtXWqiJ0U2LpNSy#c@5>9+$<UE_R|D$Tt_;IJaqGUZqn_x'
        'oL$@m<_hS$C+?!3l~R17@%QvgyVK9CX?U3-t=GjhBvM;d&1Q)g9wMU>8eo4j}R>Z;rR}!c%rqv5TIHzoUD1e^1-'
        'yd)PAUQQerbnuuM7huVtQd-ugUii!3f3>K@bJZyH@m}CiM*9!vvQ7&W!WVy~E(5k2!w%Em(5hr7=9QC$sHZ#YI67'
        '&C-U42gf)3fKbZX6;avqq_N2#10(MfgwLi7Ya5)<M(+HpuF|p<iAs=F|$c+-'
        'U~^ZS^@I`gJ`(I~Jgw>jS!H0lN1QfbLm<?p+tqaWWl|5to5`!oIBmdM60f#5Cg-'
        '8H4<MJnlvE+ZxDQ?G=Agw~m=J1c=IZJ#9xF|7y9r{o=ixg564`v#0CuhW5~FTH6(QEU+IJLZf;^Q7b;XW?^-'
        '>^}^*kZndg{4(;kh?4T^)rVhX8st(sw<$j&-'
        'ht;1@WZHK44Y|ibg?T=WP1XnT0ecsJZQwJa%mcC^a=|j4i!^MOujX}e%jW4B>EG}#v*XP3laooRtvH=X)Z0^-'
        'v}@czoSRx@c6iK!G}zkcNR7Q{I}jHDLsI~{+HU17FAEf4-!U0ydksuu{Aviam&{vT*wvMUA&g?-uq-Uw)16aD7I$'
        '&>rpmX@XaiM+mBs^6^{|Cy1QdQmp@|MhE!ZYQr!}9_aC7poF=Qax9atD09@u?&N~Pnj!;M?Mp6K`sS=g$Bkn|klf'
        'SQ|1RYrB<J_@Ox{JlnP)u^mfS@-'
        'jkE;9uU^Fln*n)|Al?7643p}{0SVQv=#))q57)G1OGO6E(_0i)!&1XXOFM%txMIV-'
        '0HYPK2^p)(Ft{edh`PSSyB%8&?LtK5v-JF}xOUZ9OMM+y{~&yUgb%UND73W*Jxl>h#J{$E@D-&do>-'
        '9RKR1NgtK{%kmBNUd@Fs7D5<PNK4nw+3ZHY>-$-'
        'h|Kl?TXD)UH)Ph8bA=La<uBs9pi%=rh~KK|j+7d1^NfyFn`O#Z)g7K<*Z+yU;T<!%b-'
        'VV~9#~{Oz^nqg9jKIk%Dlonl&<P(pZY$9ov7b-Vd!*JQ7YFDb7rY^5-'
        '>_#X<)f%MceH|O9id1juDfVt@TJGBcqI_*Kr7pE?|M8{7gJ9eVmTb#EnKL^tTs2{zrBU{BwLn0BvwH^aBwd`RZ<`'
        'G#Ngx<(zoKdk$yiyzi0Dl1vMXH}E&6$f7M6biUUFSS>=Ug=dZ6v?KjAG!{}}YWJLMeJl;k{FDspTz?s7(~&-'
        'P(7#^+w3Z$tQM|^UeCF-Wcm1yYOT#d<w$~JpBP3<I32&gXG;poKwV^-|dX+x2)*0QWfEFJ?JJf-'
        't@?5J24k6e@NUi+j#|`I-&^dJ=3f7_H8pwU3^G0Cd+Z$=*pM2`jw(H)!TQ(<lJ-cpRuZjI9m|OeE_Rd7-'
        'aQtyk>oZI>;?f*Xf%8Q^DS70)+e4VFLM^|lZCZ(d6S?24e=kRX*ExnpRi4`X!MOPYLg2otfrqTsYxB)FQQ*|)>=M'
        'OZQ81G4MQiPBF|wE;%PLX|u6zRuav9ur+z=3Ugb-;k92W?cPJRNK!SvT_IY4|O?*&TN@nzf=L|2Xd6-'
        '4#sjtZhaW!n4f-'
        'U&cO!y*IqO9VWNoF|*QCp5TMw=AUM=Ic@(f@Gi4>_e7Gid{VHiPs$0R9Xg}Y2sJ9&NXi6@oH__04?DZpAA!5m+V!'
        '47hSS*5Di>Db12?7i$Sv=)4UhL2T44sqd~|)ncqw1$9g=zlk~S^?Z<pOx)r+v;Ma|z{?mV6Br<a+^n1|n2={jLr~'
        'e{??CrZ~r?xlu9^2jqYc8;@?OA(sEm+E)VJdif+owNqoy3)!p;kJ{JzbnFCiGTCbV4T$QMAcLHMJOAhHk0rn=Jh+'
        'YkZx8uAuQ@3iV~FzBIZ?)>B4w1QVCdyb%82OyN-vHaK3|k-IC+5B-'
        '9#SV$lJD{JIVZ$;MG>TRrMq7YeVfezSXd@^WtPnhG|+L=Y>kXRSC@36@Z!-'
        '*%OY$<by?$WiNVSxz>ADqD1(zcTwp{)4<KEpYbwNVvUM=K}In04i7vFDGxXT&a9#UI4aHP=$NgBM=Th#AyZqC}*>'
        'SPz@q8^oQzS_asz@^UMF<8wLebvMj6<<vST;ya6nYj2t)S@>5>yuiL&;w7K;Qpt4m1x|OPz+Do44e*TS7=6wwr1d'
        'K0vwQ)&Xg?dzz06`s^BX5rh6zx20+8yGgSk~dDlJ&(YKoxJOJa#Mf_~gp7r_I>a?;QmL-'
        'g*sHrK};+ZZ7du#2f8G~9*r%p5&Ic}!+yj8FzfE8>?UJx$Tghz&7HrXzH(13Z$VOffKFftTY^`pHa-cy!8u5PchC'
        '^msT2^@(s0;tVi^*&=+B$q<zc^xrAb1m2SG;{MNni=+|nCi@J>He)#^g{6d!F)s|vLyp;57iZAn+Z4TKrYF))S*i'
        'ho-ac9Obc!?BujI|Il9T0QzTRzSdkbdP_L2$;Umhu_H^`%A7R`_aqxbNYAzxk7%p&V>Lsa#a+Ub@y?f}~!=q?Wr7'
        '?{I#_oR-6(!&#$rz9MeR%5rPbn6%~3wqWTzVPf9CqzG3fxRayK4L0%_R(DEmCo&HAX$rip;bv~Xx|QdE6a*<B-'
        'jxp`;<?-'
        'P}(s`t0oT_wxQqD7!AsOgTJP1D%_>e`1_&P9!+8ZTD2bDFuBh)!#Ce?QDyqP<f&*YUAHZDOCEXTq|=M>?7`>?JZL'
        'i6x`^p)`{9J0NK}v~A;+0#y=&)&*l>c7@3q*7u~Y4edAuK1pU`NZH9VjRHr~Xum>4=Ket0Rg6iT$%j^bvOZOO2+u'
        '(@tz(im?tqn34iL9W@3Lr1cjomq8SE7EcXbBjcLM3Au`=Nj1E#QIOCM|)Vfgd!fU1}%E;$a$lXy%^^^9IM+xe3rE'
        'dB>@RZ$+HT9S#y?^PMOzpc(!iE1bXI^pJ6XUjr{o~y{E%3m1N1F9HOy`W-'
        'rBBgXv`Aomi=o<m(p5N3vD)YJZ;&zh`jx1Q~RmC8)Z%a43h(Ib|!5{E%U!3I|^D#!w{@VPW6E>Jw%qRPtaHv{+M>'
        'U{h85#x;JXk|cVI0!OXmLM1X}FoAMFs~WI!U*L=uL)p+DuW&v&PsKV7bY&;mm%QO_&=9Y=twG6opirUY81so)>&r'
        'lD`WPgm^Mo8g>^n}i?xwtLr!03y{ETb_A2Dvr+e9S&lnq1*TN$}92X)Vi+wYN^wO#kfO)y^%`4se0)UibmUtj4hV'
        'p8Q+*j|YFuXABCJ{iW}<j7+F=cUK}jf;s<9WiAA*ZwM;fs(ACz>r~9T5eL->WV_sO_&v&rjt<@lRYS&pg-'
        '8mA&kj8;zxSDStNj=VAa1vNEJ^0AS~c?P^97-'
        '&Y4YO0cLYDWw(b?W}q8(qFZg103y(iYsR6Ld~y0{@4%~qfBzq#+U`lshmvkoOv)3>???t-'
        'Fwx{iT^xOk_Xn~QI`JKaJ(8;4$BUu^n?$YkKGMECjHa;0K-'
        'FNAk8H#C?K!fEePh^sfqdxkGg(iFI?(CI>i8W3?K8RnNV%u(?fl*Y8nyT_?eG7`e=%phcHGfo(whM`w?eQ)&;I?t'
        'v%!Z6c{C{hd4aJoXmYHd)pbxUGOvfra4oYFgo+mmpvpQ67RFgL8L%enk#E|KHHd~CvC&K{hO%}fE;Cp+iYDM|xy^'
        '0v#%5oMbB1kS>WYUJ%NafXvEAV*x&l+6b-'
        'wD2r<H}eT^$(yMxbjqHSR$gX&0b1+t&dNhrlwZ?a<Y%>!T?Ro(e8EjEiR{89=@!P7+puz9mqjtW+(?aJDGnq@E_j'
        'Q(;2;Ly2iIGJKJ-'
        'I~JaVGg45nE?T?N0!4T)vZW$I9ZoV(dq*aJ3)>QC0;5ihMNnAmh*&SO`Kc6Q<!;m`)?9*wgRU_I|D1P(cx=fUz`('
        '_7pJA$sUhHI=m7||zAK<`un&+cP2{A-Bo8tW9Q%mTni8AaAAN-}kD)!7K>o-'
        '2)z$p*kEcA>pGKedi^6)MA6}qHyVzJ&f3d8CL@4MV|hYakW<?n036*F(3t&3t@)5)!xPK4L4>1usG!07i!fgaUSO'
        'W5i{{b@_|z|GdkrtPt9uD<R?q_)Osxaz~Tq62-sW<aiG()Ee;<LFEO&wpzaJJ?JvN!pjL7Ba-'
        'p=gd{8xjeriZq=92#Le{`w8XB%OhffP((8Sr*BJUcO^o{XgzDDJ51Li~^p4O5T{6Gfz3&G4uJ?pyqxai7uF&3C>s'
        '_J4F3{fPN<-&l;grl8dbf4<)yqa;t)}gYJ9}YH@UQIIcl*Yk6@0!6dv-'
        '^*E)RO>?phGzo3}4sy&pXN@$uXDgCC#2_~FO*(D6=RzEow^dHwRu<F_x~y?$jX-'
        'oCRPs`%<X=BNG3i)ZhDgi71r-HZMh-MbUrjJ9`rx9>Yw=K+6CJtw1lxIhUXxccwWvkl!DwCIi(1&BzDL@}NKQ*tn'
        'PNUv`8iWI$4Tkh3hGMP}1^7go<HxV*vx;%`{ld%el?iwqQ9D{$E6M{rSyf@K>&jM8JVl|T}8Mwml$N?+!pB;#+{z'
        'I8fol-'
        'r!4|ahPhhmV}T4>t3s*3w|*9lTnBD}7pNyL`uy24;FAEL)_9JBmr($}*Uzn8Cc1d6`x_jmh!D*yBQr!E565nj5xn'
        '#8QgK~|g%Fn8e!jXYmp@suOF9x7-9P~b=H-aOyWF@v%kM|&k6C5N^@=uv)sr3Ykej5BL-X&HF7An%QYRGFG-p_FS'
        '2&HFo9h7<Kt+{crft7q&U7s-hXtyOE&linS;^wabev~J-'
        'whW~_9nv6)kC>ZXOqkQLx%6?qXIM3&efr!c+hyd%h;hRYK@OSAB?$C3xuosakZ30}>=*pEi3_Pwnw66))gG6L}K='
        '_rt(n0f32SwD_0I~HTX$6F)<-'
        '<Ed1_D$qvPpd#ZOKPm1xd^ITbVJPC>9wqUbu=$X*;Fz`CNfmKq?8)!eNJJW5m~deb(nZ(@BlP9#)o^pr8RGCTind'
        'YT^az#LF*bcSh+U=^rLo@Tc$8k6%MfxhmxSRQTHLMRr4W&nPahyX9#*nZ)*;Ooc$UZdZK4ZcxzaK1E(nLcSAS$?O'
        '~fDc+IzuDk{J9pOrt!c;u!<gODB`LNq7!Sq<%NUVR#<N`>VoD8z*T+4O&(Qhj{W(pv~V_jPjCQ2P)psP!U>zXHH*'
        'e9yBKRa!%KT$ObW(UHM5Xp=|-'
        '{cg<Ruj<kMiYUKYM7qW<~J0I^hqBDu(WK4c9RLq@fJ`nmG}ptq1XjRV(|3_F3uQ=ELc|knl?;rb%&);uv5<4ur!#'
        'e#sYm%dEVnCEIAV4A@ucmTA&a%UtXM{aKf{<`Sw=amQub;R?~V-'
        'R`e6fvwGhNI=Ub57!|E6*%c?+Fype5;r4XaTWr5-'
        '3pfvMVt~wS{_w_wo5%TRiT|F?&n6FV#BqG%#*M)M)Xu>GHR4#HIKa$u#YK_9NC#LoKD_b&0y5el3I'
    ),
    '_portable_underwriter_780ea478389c.reporting._core': (
        'c-'
        'qx{+mhQxvfw+v0>gbFBsQg%<+0sIbl9Vj=QxgKdnNgt6VbuIph>VB775S**y?sq&u{Jfe%a=wE?F0VY}w=8otUti'
        'Kvh;&R#x6BD={9AU)QT*b5a%iuE{q^etFpxmw8ttO|h$+uB<MTbyIJXeA5*9>h`2v)=jZWcJQBa*_CzG&c@^MXhe'
        ';&Y`yRHO_60uxy9~DUR8CMQ@hbfJ=^Bp6}Mm3n@yph<QGfb>2<!_!NAkxZSl*#sFnq{UgcfB+~jRrw7R8vWLgwmx'
        'h?eTpWeNknupwE2ZL|Q3*F*JgG+adW9sK$SGUvTyRs{QC<gIrzuny?d7D%_{b-k0D|i6^-'
        '>oze@I`*HDYAXFDw+=sLdib+{ArOt|LpTmKL27lGqjyKZLYfQMkCM^_#$}yTZh=z*TuG|Iz5bCNAoJ%<+qzUUwy<'
        '-#dW#D_0rHMqXho<`s@GBUVr`V_t~p&zIpZXUG}#(-~KiG@rQ4}fA@B38od4K%{O1ae3gCu-`~D<UcOQ=-'
        'ykovoqBkaFT3yRw$*PHpuDVJt=GU(-gD$}_t6_d`ej{p#f<{0xDf~et+T>3Pe-'
        'F4n;J;4eRsPnlDX!K^YOZ22E83mlW|*I7eF2Q`(ocz*|KiC@nZC{-'
        't4#456yq(8<+reZ@VV>hvDt>H4wX7z>{h>17>QP{1$%fR<pMNu4tkEc9%D0TUS3-'
        'Wf%4bRvqX)%6@$F!(YC8^*a0Ot8c#k>AQD8-2Zy@=1gz0bAUEY6b|#`jwm4}c9xvZo-'
        '%#x49}8h=AoS`HvM!uy0=5Vef9e5@85m<@~t<V^8@~~=oy+H(c`9QcOWJTcshQ$$;)jbh;N%&e#VAB@-'
        'L8A%l<CcxIP5;)y=MML43)#%YAcQWH&VIe!Jg*EW0j}LbXiVt31~J)@x70pDgP=>^#}!KBVm6oGp(XJLOmpJ7b1Q'
        'cpaW}-'
        'z1KhcN?F!<ei>*{{@;P)Z~aMU&I>!k?R}I^L{iMk%mfsvhtIK&#R`Xo5>p>Cuk@7k)APK#^X16*}{ZAToqN4Ctb5'
        '|yAROi>Q?FHWJ#JexhOz{*Q@=qSg{@+ePwjw1O)i6MKyod?2EJyi63^Pe$BvtE!*7&WC-XrV+=rpWO-GWOAue{x='
        'HFvn+H55cJ0b;Q=E}(n&Xm`OVp+$N}=s_S$r5%yB$l?4Ad<&cjc<!2Zs5ffR!yk1$`)*GZs=H+`h;0=kyOZt9RM^'
        'v!tw0IG#SEXRCUTI%`v|yX>Nb`RUhBS=7@ix-'
        'a<6s)Q+&7kg5PvcWUez^KUx*rp(Ufb9l2!<(Qp`~uCv`qpjnxkAhc`|P5wJHRF1WkBmqxdbyNfER7v0@Vy+pFR76'
        '(JrfUyWeKZVzbHG^4G$G;ssn4Yp~yd?Yj(`c3C!Qi_JPsPQD~cpP4N{D~)OhUeD;q3`w?|q^3!^PLMg7C(Ki#5b|'
        '|Gs*u@#%;2`b;K<BXU7Zx$9mp#O8jU4chv1i0(xcrDjOjqqLmTBVH9~Wm0JmGFv?virY8rsJsyElgDs%XC82$NNY'
        'C76&4D(NORJeyGV7jV(*d}{u@nc=KzzVRG*JX}o*u=qTewrSlDxzgHhKtp+!&vE_4pM*1x7)lIrDsQ>WS4+&!nFq'
        '?B`*|qFOEUZz;MNiS$^c<Ww*J7zp+7;oYGj&1b!&{svO;mN}0AsEszWndV(JMLdbt_rBKk*mqLC6M+*5b12N<`2*'
        'r?Sk-'
        'C(;Mj8UEUl!oUbqz^{DUl&{+9?5p5{33Z(h1l`Q6?!iupXyQ4<#ka7N*c4b4+NgDGH_(cV74ggxrQ+F;*}1s4mJ;'
        '23U{p5T|>y;CV&j_!4-aUMHYgHbo9^?i9%TaT;KYb3%R$2~51gNPRiv@EnC`0N&yZ-ZQfB-M|n-'
        'h({Jaz&zrK9ffrWJC1e)zC_t7J1t)i=tL<#++FE$m+itV=CpO$O*SQ-'
        '^V^CR$mLIoisIP|G$_)(Jrz(dpzrv=qz!3*Et(n;>fvU<V(b97IH8#^NMdX-'
        '>ffyRI=WZ+h1se!&Wq%rfbPY?vJsgF6`7>hg`TggKn6L0<{DT+2f3lITa+{6jZsG(%?;zQ`vIYaT8dzP1-'
        'f%rG~KPS>EhOjHSWl&wg{o`QFQ1>bywsE0mGG@QnWBVe5&;G3F&7TBj0b>6Yv$et7*&Dvm|-'
        'D(_nEs?mRFD79+1Cp+5;nOCRnEy(wCli1CDeYK!$=Mnp|?i-'
        '!ir%r|v)*}_vI%fG*S{avyy!5%hVQN8bWdyJ@cS7#h+Bvo0UtpUdTZZ-QWp!iJ#@|pbuikl(Rc@Y!Yl4;P?AKDBI'
        'EH?G&5z2z>Ls4E{bs4=bFDs1Ja*IRs$bbV&43@6GtpI7|60UK?2&N$D_xa{bgq@UoXX3)%3~WQjpYaR;tcx5Y$k4'
        'mn?=}VZoy}$o-'
        '{0Azld=T=vJGH(t=i)$L{89>0(M#5WY@*A)6zLY2)3C>cb+9~n2%po)CH$7qP=GEWa7d;32hu`E(8QLVdbnG9ztW'
        'rD7G;<@*l;mXYg=i+<sf|4`4ygk~@O=r$#DXZRpVm=k#l#nHU}e?s-'
        'eNYeZMLA?LIKQ#~DM3sWq&<j0`A*c8<S9;WyJ|0d{SeuwQe5E&-'
        '`v8Y<$Dm<b9O}ovz<rQr@f;TqfBb5NtvR$JSThP$6e6yJxKjgBmx*RB7SKtgV7^Y3DhEb9SKx>Y<RyzkgXWTgl80'
        'OK`*+DVJ>EA|f>jgJ&^DXR^OmC?eM7ys{;1~n1)7XYV2u{qIV=mmbRso~lP?F;HxHt*T2qKd!;K*!XMyFW7z|xJW'
        'nP(wTCw(+ypsLkVgvfBk(ZnP%GXYvm)SHz2JoSUNoFbjWWG)yoz-'
        '}kReTcRSh&Udj(<2?6lN!+FN2Ba9{AH)9c6a4$1KeqC(M{Yj?I!$4RB(P8oz-'
        '63qH0(rTr$%IZA%}8<b)qeDQ2+tA^3<H(Kc2=P%~5xD`kY+IC-'
        '4lA<XP&@IodnG%QvVA9k9iJq#)Ko;j%?j>L9yV~&ObOs3h;A_5Us93$2C+bvKnR6uHvu*86w@gIuryz7VhIpP6Ns'
        '^o7WJ3+z&6DxRNv71@DZ@z>#Lz8#FJCQwOx_>CUE0-3#x-'
        'IDyiP`|ETf%2(RHjLEI0elNF;DDFjN)S9)f3E1zN7#cN1qf%=NjH3c`-jk5*-'
        '<1EZZ>ZWROFV;l|7~Zy3}Z?SOv;78V|--'
        'H&w4s>wgl_M0j_hSEqL;DZR?wo*fKya17#=)P%Rke6(ngYi~9m^u+)sd*siOOkX$-b#un-;J{jRC$(-'
        '?G}K)pm#C8KY+qIjsqG+K-'
        'QwU#?vEvxB|PisW(7fG7MZhNyk$Fqv(npLc|EpB#KgWOwn>+pY!TY6KAqLGhimR?fSX-Fg^7UuBZ!H+p(hhIQkrc'
        'Ap_fI%>K$$hc{rWPbUA=V1YCD+ldwDo-q@;o#Z)O>(h9oM3YyS#l)JY+O|bYvg-'
        'c)JhjvzC>YFiOmv|>yy1O&z6Z&})_v5%_rb(WeXotn3Vk&5%#|YofI$mCjKzS#gYL?82_jXUV1hKP*>rQ2I_U8zN'
        'HEqTGYXozQ}of$UXJ%%L>EBpk?-'
        'hwKEdKLpVSvW!$x#Am;n&FL*`YlQ*<yuMw58BgA{VV!NucnH4CuH%eKlVqEkwO<KvK(_M>7@t&%{D7&(ESxx%<>Z'
        'oQ35F8>t&1~<m65g@wGQ5KfJ77|<7e!x@nZ-'
        'C3xc|~!vEOuRjN6v&Q=5E2WA?@wYMWL+ef0egam?1SzIaLmzNjqF(nt~alEsgL*c68k48po3K@IbI6Q&?M7?w){c'
        '(>w4zoqCtmvJ)GHsNMcsatUj87tBNKIL9gWy*`UM@!^~W+@@YyenXS1z(eO%aWh#p^^T)b_CShN{ahGa(}-'
        'eh=Xb0=d3A(7(@vN)e`1<P+d=mhJllzL{wFrRb|>bJxU}!10js%?D3Yh8(%GfGK-zh*9JXy=gln$;<(8?ATyRUE!'
        '?wsP4(vGbv}q(x=ZB4*NGwhbY~B1)#j{!sp;c5&!?Aza=b#ied=xsdK{XeXo~WS{QxPc}Jt4QcS<%Tosr^e74hGN'
        'nz=HE}+woqutpP*hV<}Fc=L_ZbuIRLjVJ`Puv_{;1==DCGeVRP-'
        ';5kDe;tT@_30t100)5N?PrUBR?g3UpzL}Uqj1b~-'
        '1D3mdJWZZX6|n%fQ|^~dBVC!zyw1Ghuw`MT+d@T=5rpkW_B@$X)YX_&21{(J=m<R|e92VwTk(r-'
        'L&xTmo+z{mh+&rUgSjVfmj!5Rw3_f}M{y9xq2-nkj#e*I5e8A7Qht*iY-CT57!3V;es*?px^NbhV4`1!|4JS_f=z'
        'W3<tEP+>130OdXLj8wOQO`=eV*{cl`6yvjy5M?nA_tp<Fige&=ik3l}%Sscj`OhUT!=lw3HWmsW0go5|^KrgL<Eu'
        '?x8~1H3VFemZ+P71+C&rmzd$C%g+6Dif{CHx?%sNp7fFz-fve*O`B^bwgOJ_n@RV<@;j7zpK1TO{8cLVKqBISdO('
        'AxkKFQ-'
        'E|38LDpifK9~Z_c6fhTt$Exg6B|ip@lBnaKDC(;DfDvP?XF{Ac4WQ4Q@fRfB#N8`+=eFsOa?~9<P(BTj1B^+GfXo'
        '&q0rAGgoS6)#jka_nt&L%Q9kFcXfWkG1H|>80~65y4k53=9LB<oXNT5Ufwtd*E%qyU%#<>T5!SrB0)i|zbzA7ppb'
        'L<J@{2a>>dpMLI63_^<$N8a#1jo!G7h|$iV-vq6_}Qv0q~aW%1MM3%qi+wn8ZJ5SfDD=dk|)#!OU==>>_W$=cUw~'
        '`a{O~%sd50Rcbud0c=KMd7?5i>5w8YU>yY^6H%wio6s{Z{F9nsddh4>40UL)|BO`a8F?&nCblHmYsh4-'
        '0mrkLX&zHQrQih8NeEv`!Be5EaY9W>01@#W*y&V)NnH?jc3#S^hppqFuzjqft{Q|tBM-'
        '$XFG=d=w>i_EdX9=4yK(T9NySY8dTj-3x&e83ZiB3N+X4-'
        'i^TkM}1WLh#{)IweFv@Cq2jEFYnq3xPNIMyJ$nmy$G?RkgBVn^@vypSr|3fvhDX;2!m96t_xw$>iNN2eYwE@o&<U'
        '`BsGSslG_l-;9a5bXS;O!p-'
        'cz`u=Q(LOcC~B*cCR@~dfjn{<Lr^pw^~a84R1>4}vqK~l9pYKSLu3`5?el|l7QOhH=|RL5Oz&+=DWEtS8<iwvhJj'
        '|4ih!&Lgy>rn-ZqvyxCiu3-'
        '|FZQbw_|udw2#S7F4XuLU&=x9UsJ<@qUiHmQ;gg&hJQjN3<YV`8(2n*B1L#oi)#(Z9JM9Y%dwmcLbRd5EyZERKAN'
        'tOz}tIbMh`3AujJcamNj#u#tHpYrqtVbs|)Ac?s#}K8f@(j3HDKbQHgaI>{@L+|gydNZ<N`kBoqBnQ82fh<4K<Fs'
        '`VB{_cZFk3bfU&qDb$+>gYL+l?L#_p)-'
        'w?P{M7_h$*@_T^`8zx#MtQJO&$wXOT(NFNzP!aT$LqTG~S>Ey}q<w#|akvYImenUVB+N&_-'
        '!@T(48f@YSW=%nw192ru{&N%Cqa^R$#F7d<4-'
        'M2r3R2E|5v};hS<rW(S5Is|dF!ElfLR_!{j9y_S#H{~piIyHX72g<#~OIx5U%i428nc$SE~=@s=ETSgKj97*98S='
        '9nE%`@7oqcY4(0qxH>?nyJKu!CJU$G&zU~HA#4d@j$B=(IL2F%H#5);ur8C-'
        'zoNo3gC^>)zJ_Mz(9}}FZR3&zC86`cYi;)Vn3pirmcN#}xC)(&cTZ0-'
        'YhZW!l>YgY|M^1w^Tjx#vdBH@lQOaY^)1$s3+EsVI{FkfHT~yP{LdHkpD%h4?Rm_n@f2g29ylx&5bhnPpO__(x*^'
        'YgbnsqWi990ss-'
        'j2w_PV`0%jo7wyRB>7a2NZwtctch(Ksi6Q#dE@i>4|znuU`BtW_|WiW#RIyt^ta%c4=0CkDjZa&ujPWdd~BZ;E73'
        'hnaZb+Jd#R;rx6|2>@JIDj6S)BoZHVSERnc<HK+N=bNNm=9^-'
        'Ml@5Sz+yaKes0&y=FcR0_ARG%pM|gADlq)dcuImy5NO(Em?|&zma)T#&c#rplN&5G{CxBGFWauc<HlVCj&Uw}D8?'
        'xr`EO8B+xNc`W_jQTq$rQgrl$0fg7n8f1&Id8^E}v<_ix{ZUVXQgaC=8jDz7UlC9zA*nTFEG#7PefYqe)J>N8B}V'
        '51vxRlZ|$Ff`ckMggPK(GdGPKUGY}pRn6>*&e_ave*x{k&2J`pG%~SOw_l#Y?)~MnPaS-'
        '%FmaKd;>LeQ+x@do6#!<LjMxQ$Oc2%yf<<5VboR;fFVxuwyhn<14mYhf|Ap5UL*%Cp;M8WEf0;a8ScqgXR1=0yt<'
        '*r|9N?tEk(dzO!1jP?Kbt1tQv6w+_LyBXN8tr680aJF=}CKxbOMk^$>aoxpbRH%a#OC3Sw*JW41u8GqkjUTX*e*p'
        'GNLV_io7wNP>0Vk=BgwU%t#1FXU<dfk8tKN=P%45EBWRs1c1%S;c_u2Q~~u*$YwKGG4^}#T$|`*)8w(=#iY?nkB>'
        '&5G9Y1^1W3_NI!T^WjM>9NVnXd6`|U>db~{K6eSC;>d;au`KTm0xPfvj(o(pPfNwB)qCr%sax679<)>9AeV|`w^&'
        '0AQJjJsZe=}}*Te_%EeHl7pkU>}?AM1i)|N2q+4!JL=~!x?~OnvHld2=e*q0*J<cK3ypBP+o3xO#;h)Pb>^|%CIW'
        ';ty%(5?no?mq9+MH#>HF~<!17nR@*|?8O-&1-'
        '4>mj%67ugr+Spf2r9U*p_Ct=*WT=Kw|&Yd3`!ho&`+FUW#1=r^_OrO)*KRGjmnNnU0q|k#e|oPdrT?maX!Z8$~$6'
        '(jbfbFMnXDQQ#rG9a>%7F2o+Vwx#v6=wM=+C2_4j$ovD@jAIX0d#V#qTdVhJP?rL#n$lmBlZACETd~?FG;nx}yeX'
        'fg&N^~e_d9p4(fD;cGY-M(53#<5^?}*`y5(cr|FRv2dhoYIu6Ud%DfLtQP(9>tn)j~LgR_+-'
        '%9*PiCkEVhPr^%*RcXQhP4W+rDuHrKW(CSiA+5;!!&)nFAOKR;LBxsy~JMg7<<Y-'
        '_pd@F496IUG=KcDl;k2?Q<==o%yGufXNU>!yg5l}WLhjDInfcc|OGtFOVltF0aPG>fg!$?LP(Q3M(Q{<kbQiZrWF'
        'm<B!E}QvNO-DiO=FayclQDXY)?d6g7w@PZf+|1uJNf_U&i6U}R5=U0UXJ{tH+gse+r9pOVN6`P>l)uhpX0u%o1*#'
        '^<`dg95hqlkFnWj|d%%C?H3>owCa2Fi^`4FfJlOGNjx{d+znBc=nI4!;U-'
        '<hWq3?Sq@qjrlJA%g!pATymyC|yV)i!V5k6r#cj<*Ju;_$0@EUR|~`dCR2{wNmuzk{>KF%Htq1kNgt4n$?*hv>)X'
        'O1uOeymIm#qWP4nRtXGNg4S*dcCnR$0~6LjPvwb-HXYRQs8{odrzG+SS#?E*k`<ZE<t{b$m*|8FlVsDOO%nkUNqt'
        'wlz~&IoRrZaUfp`a2CS~Y6=0Q?6p*QvA>29K)oTyj+0RJ2s;h+Ac6G^aVQGzju7^MIOgH()=sU2gWv=Y|EtBOhwb'
        '&|M;*aJkiDb;xO4;Qv+P)-$-'
        'Rf+jA^D!qVk5MYoeQNW>yK7MGw?&h8bz>Zt?dilI7SJAvFW%Ba8w6<b<@<tI%|LyUNND!&6Y}sys!NgUbDxV=FZ2'
        'ua|DQ*0T48H%A{dWD?~A&gV}phBULnVC9E(o>&-d2|Vt9o6*qH?fSFjgX0-;XR<P+SWe-=Atu(y2lNgUvF1o-'
        'T^2QlqqTGQQZfT>wG1h1$O$>E9IH*_j{#S3~V@d?9%YlZhACL$}JadTZ}j^zd6Z!pJ;EWd0Ds>k)wqV|}q$-'
        'YQ8IOkZ6#Ds;vE3pb@g{Pzx^2BN5H5pOC_hg+@fkw*SHMLt5yW;3s0W7ug9TgDcLH6XSazh>x{dJSrWUJ&WEx+9;'
        'Ogf4tg?L@f`)F@9s+LklM+t1aQFK!a_r$|N0uEx#1~0osB@XF<IUWPg!!-_aS@(l-'
        'Fp~Ob2q3ud^|q{(6!xe8I5vmJW+WAAbm>U(-'
        'e=JerWh=mvJ!A4wb9C7Z4MF5pxY2EbOd3ZB+fV|nyphuTrjk$2TVVGYWpZL0j6Jt<|ZgK5MbzhOicA2kz(Lvfzk='
        'I<KS{dYQBjXtKU)MLB>X>csmKI5fd(U>h0uVD*;21B9Wi3h2uFOf3EUp#(cdBp3to-nAIW@XhcXdQ{RQ8u5up*X*'
        'fld!bMY$9HL`?&fDlf8mgVa5jm)qa8oB7(w}tTl0VEoK$1u%h+hm3C@bO=CRlI;k&nm`JSKHl((F*mu1_+rR2}WF'
        'Oz1QWyGgz4U>H-$a-'
        'f(W*B2caMr=5L;sF_h<5k82eHG3Ai)Q}$a`GdR`#vhM&yQ$?s{?b0ESVRhau$g)!{mdbau9Y+d7UqB!K+{Ymdk?&'
        'tBMT=h~uS|d;Dc<wdqtaQ)`@Jxa1zG6N%nixyE2qrmHz{<D(?3k6^FvMQ4}Ny$Mk6PeSzl1A1|qr(zk-xaJ(d?*U'
        '$Ypqc?KyrJ5XxK<a+qgMl?r{uiG{z<utv$(^n+BjFxE5W#sIQq;lfbE^zjkW`yAps#Gm;W99JwHI;b4>#pZo%|cn'
        'HTT?#V;eWh{-!lA<>sX^RB8ZB-'
        '%tE_M&guL_nX4Kp~_wT5{1`wd63?s<=_5HK>m{&eRnyxWU}7<Jn;ziMqXw^~S|nf$W92@O<eG;2;eC_r!ta-'
        'x9#1tGJut623h*&=w2nff!=YP3Z?fdfW%8Zx&n!z@qzN?hZ;a*ua)XL>t9`>;v^cj%gVwn<l{p-'
        'hqp|!R4)@%|H`n%-'
        '8mc6iy;?Ys;w)*RX``Z$v^7&(WMKUUqNcAe?K|xmu5srifsDZ?UVaWUFqXDsl&J!oKcRGVW!cMcSkDobi6P9qdE*'
        's=cfR{u|G@{_wkKd_B15dj$S#^ZkJ%)Z_1$sH^Mw(V1d+PM7wFC!7$zpIs(x;5(;4)|p69gFX%va4nT`i@MM55u>'
        'O|A(lfQEQ*Lf;lr!USEB|I`kih@#W0pv(JJ@qT8OpljeX3WMax>2R}Yjuvt10m4)y^#9p!Pq{>+}-'
        '4TYy)o5O7!z1|)>KZ@EAuEnf#m<CRqV5@<mz3i^$7S)*BqRE(D-0FHruIs|=)Qj}P5bs6a?2@qk@Jzy-'
        '<p=F1_o4|i$RAFLzU}#Y(&ET{8lwm6y+4{L@sxTP2y}fbZy_u3&=b7yykvP3vhTKd|GhV7#A9x_{lV)MN=IBTvqW'
        '5oB_`Nh;yJM<Ql_yX;@ZdIiH{mk&NVW}%c7EJehrvl^>(I8&u2|_IdRPoS7{jxNGec*WDGseO;V`>^yw2d`b^c{Z'
        'zoTuLK{pIWGk`Se(|||rF}`WVHEUflv~mmBEH$8z<wn)O`NgJiiTzULgwLM*0~|eqW=oEnA}%^PEkRzCm#9-'
        '?w&}LV>&$OuAy706%r_vXpCGsKn^qH$I`=5D*-VqN)dvXZ#V4n5_1MA-'
        '467cNa@q3)tM+UFS0N7IY#XDqp2(Eby_2h{eU_x113`>P7p7(Vrn%}u5?7oXz9(OIOJCAl&!Aws>?Z*oI#%Zguq;'
        'F^~xpS!p)1HnrwI2u{!oNu_MssjHi4%;|p^=Bdl1$ac=dCGw???tnXkD*`mNYVgj+e#d}A}`LpfW^>kO<8G1esS!'
        'NXSHBh127rm(%q07}v<nRPB(m7Z>FL&ZHqw_SC6I)+SWCn-mociH>3N7|Tj-'
        'x|7I9>2zw$Fj&!oYo=hhL%@FfQyG@IH!F8gZ5>?OUdP0RB>?k%As5%8>9wQ6>-9X*hZevdc-'
        '~Iv(<My;0LquLVhSpc3F7QDYG)$A%2vxTEA%zPTlv)V=8*0K1D<*#*wQ%e7Jd5ycM1XUE(+Rp+spnWWzh+_nR$wk'
        'lCn?VmwN1G`LJGgfAW5x@Q5q7X10P8KwYDFwIUi7T~w?$bVcbk}=lR~E<4?kG5WADOx+aNP?DKSYK7{Y3OaVOc~%'
        'T{Cup{rI%8s{1C?f1kO3s-'
        'liXg)yG!B25OnN5zcm6JzqCyXD;XgYeVdxA*;*v*niWZS_!5Ap3=a^_ZR{Oz!TFz4(1UAahiiKuq6J(q~4GE^a3z'
        'PgLb(RRL1nj8s4KmX91%U1f3i%NRnG-!fb~%XQK4m>4-'
        '*;Wg{(jPCZ2Rk;s84K&D5VKa+c+w_!h^%1pbNua~rt^_R}lId~q$s(l-cKzK{Q3I-'
        '5ua9&>Ao^zW03biwsDKXgzdZvQvP;hU@luDEy@L!PY!pu{kD{k_#-GrZeY`0800@;#bmp-'
        '`pwkwqz6*B<Vv|?z@yju+%PxzWs|V_0L6m}lw<iY|!Nn$~8a;$QirS_yxU8x|109sY4uY-'
        '%BPU|U^|?J;@jkY}<o})=3plVL2SE=JA!0E|Wnoxj5O2LwgXYI#^l+VXG$YI0t+N9fSf2hwDXnafV|mOoJxq{pz~'
        '~szvi*}BvN;ah9lf^4!<XA%_F;==aS&E|u!Y_jfe$m>^?vZ}ac0XQLt<cq-'
        '5&_w8EUCY0mEju+v3P*fsAq=7$V{<N(4iea1U7faTdYB0i+MF$T7a59bwm$+q}6oWxx;O?;+<ovRt?0=q2cn3p^O'
        'FkggHTVoN@NkQ3H&F?uoYNB1le_W4jW{4(ROGR?h!S<vMQglW*npXAU$&JYfMgJUHF6C3$_cHB%4VjZ390s6QzJr'
        '3o_T#q4<xp~G2f_~cT?C9gOYMpVhHv8;pP)Sp^c|fkM(+|XnAPwqf`F4oe)qy~!!qw)tj$8L09K;V0LSgUoLD+)!'
        '>)h_z>G0-sQI-WtJ>;XcjTxn7sS4tbzg2xOMHWMctFg!L;fQn{bPy_j2G$|VeX`E4>wPl}p3#1B40xZa{R6ZI34o'
        'c^n|+I)H*_Q1&apv6*3M(KZ!h?MbXaW9b31!Y(Xq29p_#Rz#xe13EbDP^_|TJ1m?G{!>**T`Dd-'
        '$N?CISnLy$f_uX-T)7XbS0S^q$9P%rlz)Zx)zOpQ(-'
        '5*LiPKv?Nk|9yytrgypBHTAVl_U|{x2jgD@SHnJ{tg(^G)+&5**zP!BB9C2TXK=0uyt)`q^jde*(Xj#6D6~LMO#f'
        'wJ)40`ph^!Hp%v4N1#Eq^h;GmgQpa*)9gqOXzCR5*bVJS_8QTflFma&ub@mcgheR|Z_6<d4A;J=yn9vpes6R}7}-'
        '=)Q<#~jrd#_+dy*5+R9Xd3Y3Kv2icA)UKF`@2z&^nBO9#<7@rUqNbp*UNLN=AOpwzpbPK*%PV<edc}fT$h&aNrUD'
        'x%-PSlpm=7j!aS)LJSWhR(7CC$Yyr?A(HV|HgTue(3+>`-'
        'rqs%Q20OUVbBEsueqY?CXA(9ht1)Wu3d~dZiP#Y^^Oe2+(U1_yKR6F19P&ImkTb@jAGY|6T>#mqj;T<U4#K1KuyT'
        '0>JVbI38fV4@fK>TTqx@8rE_$4L@-zw;%Y6R>(}3W=nC(Mf_Eu1Rds~ia>a?iufbxjZ4=6deq=6~8spdP-'
        '{&hnrxpyDz*MOYM2d`0Xh`S2jT>-gudT$CijcmG%`nWD0L+P*EW9jE@ehD<)ujmIKKul#_>eZTB72Q?6q9$C>-'
        'hKtfin^)#ThEO`r}}e6LAfwoRj2vP(kku@1*tf+>oFYG2WQ4|VOj5P!DK3S;!{%!1$U<UbHpsEyjxE2=-h3-'
        '@W1M!M5QGOwqP9a)1PNky$XvN9x|$LwR#uq+$GUVwdgCPbk3EO=(3;J$f?8cm2aXFGwL+cs~^=b$+vk4wiA&cYYY'
        '6c-9%l-G|!Qt%jy;Vtr^+9(|_Ex7+b%N;nFQnGs;Z;6I)qx&ym}lkLkg9p^Kg3cd3npBOc%e#I8J+D_L4Fdw>ot-'
        'mosN8TM222G>y~Ry)E-p&?GUEkmh(-'
        'kel@NXeeMVRS!4KG+L3(`5}ZIo}XuHJ?_0C6_ej77V3gr9q1?B%{E_Btm|19$Hxo5c!kv{w3;Nunu4)QfG!f&G_@'
        '>I7;f@BQ$HQNP)Wr8}@x9H$qAE*YRB+#3%OK2KuU%{dPL{Pt!E013I_Hk3ui?KCvt5tuI7d6j&{WNSLeBSi})`Im'
        'R6>@}@<J!SlkX0sW!MPV#W~KH&`OdRO>Wo*=^2x3RN;AN-FQX#AlQ=lkjp^1^D5{fq4sOp367$m=&^jc$NA0XZzW'
        '9ciquz1u>iI+54S#Sqjq{oP2^128ZP9+_HX@guFy%_k-'
        'v#0TzLXr|Dhrr{SZqC?Bsd3`Jw_0%X3#S*FVsVM2^^W?}6nD7@hG>u6wMll&LOXJL%eM`R=o=m5QF`4cH(>|^S#='
        'kva{O|xUlk(vcNp1}Nr+6EDP(-{_%P5zsEw-4Ly_}(!jOB308SeT-e!yd_8};-'
        'REm2ArRcCx)sp923$<J~m#~?b3=xBZ=anqkfaxnA0i8Kxr2-'
        'eo{pLy+#JX3AWZ(TkzV#f5%F=of~nkdRh^Y*iPYHK@%3qhB$DvlVUc%`au<Ivl5cVbs|8zi6_z}u(!P1*82Y!m=u'
        'e|hbiLX{|FGZY$Q>8DVPSXkc^qRG1K(15V`dYuTI?ez^e??(jeqaqqE9KX?h@Y~Gbnk1EjfwFG?&B{U09jl^UHsy'
        '}mp~q0%bV+#2ZH&2y`xd}rb#mBBXhg8mBCke`P+$CvBh1FbABq`TT~L`_Np;3B^RA$P__6CdzEewlpZxFtl_*Gwf'
        'Hb)DZ;BQcsFfAj5A%R4)FMpq<3TLE^*;iQehvsn<ImCX32YOBp7KY=^#Vp#-ZJ7r#`!Jlo-ddC?S7L}VJ55$sak#'
        'P>I-<Pf~JBu$OEsUZNg=LdSL5+edOrg+Q*0QU+%tPaDA|{KNSwDs|3r!aCyh}|CRMN9}&t-'
        'HsZ%Fb+O%*?8F(Mi3K{m-'
        'lnVG8^6PrfAtW2d=4;udwcY=!@H}JJm|a{rU`#4Dv;zX3u3}wvkQw6Ue4dl9ibWAx4!<aIJ9XGZCH1`j<XhLgq&k'
        'rAKJWnrFdObfl0eT4U+VkEZm|;kaWm2JHA-Vk>S*bhoPd!5*N3bTF#?CrTDQ?QnE#qnI{+47DhCE#oNf&8C<-'
        '9FFJLLd}6@g-'
        'KSpYvNz!8oORhYzuPT7XWD=^?^C=RZKn`SQS&;5GZ@=*>j`lMNPes!5ma+)=w1fisD_w^mnL8OrMrUI+bFyYg@id'
        '#%cVU8EeHP8+$UE#B0^D>uEa>Toj|^9uowb=@8T>X!X78Mjb}gC_-Qhbc*RPf8~dua7EXq7EhW#@zY9rr@zoFYq('
        '!=jw-'
        '$iey>w8VbMCd~OS6r|OMCnl|6qwPN0A@(IW$9<xye~*WrKM@5eBY<WLv6NJ?^n=8FP;}b(bkG+ySbe#~fP+rtjEl'
        'E6_%YWb@F$#%~>F`ArdK!{ccaDvtC_R_jysmuvuozUBnYWQiC#q!}QrwGoX{MDf-UE|c=+K9^zeXz)ZwqkF*X=&2'
        '80#K#B0>xvmBln`)<+d?h?J;_#Oep%J@ago4QiJ}~Ii>C(mweGzZ7iN)2WY+)>5j?*eYdwySX7ppKRJ?vjKkj+g='
        'd4pCP6TTqBV2h?fo_e`Prpk&g)T*cmIOy>>t6Aoh2jyKcVFe#dAZ@d124@0t4NDL@b7$-'
        'X=m+~Az<u1Jsajjajw$Q0yWI@vc+{ovxwCoF%1NOn+PJPIRo1*xJRhICxsTMASBqXIVVse^cnxU4=zILJO&RvF=n'
        'uR5Zp4~=`U@g6DDUfj-'
        'P0XuU_#VfT9=$9*bX^jW71)X2r>mD>fT4_B5CR>xPy~p1eE8rmj!^l1_hrX#P`F{}OkJy#3&|RvXbdScyZ_`Z``R'
        'oxb)5Jk(vCsl<pMbNw;Uqc)~GL{X<DZpRD{8jH4*W9yD@_!WK8BaIdy#_8w>KVki?$Vl~0k?JS=QHM9Q6hH7AzZ1'
        '!kpz#|{o6x0+{|jH?HxG-=`UHIwFx|k7u@0v4E6LAfHc-AJ)6-'
        'Tarq3wiw^S9PfQSAIxV#T(wddI2Lc{x7c{v$BvKwvPIEIf(4D%Dh1XIn$-'
        '&1kcE7)l6{IZQSBhB?6_LaACbMLqKwk%I@?$bq9h}U7-dBQ|_$-lix>l%7-'
        '3iMQFkc0>??@he}xxN~Mh7g`b+)EF>TRcCmAc}vkIzO1p7<yGcd1nJ|)~=~Rv~4r0NfvaJZ>M49L{2RXdYK2h8oo'
        '$AOx=xPbm|<lVK(Smx<Y?4k{Lh12%>P4?hAYA?mm2O<c}O#mR@m`tyNQJzTUBY99MkULI|c9DAdqWkfrZDi|;-'
        'tRU0XNI`ri;du^t-YDr)D{|7Cwd-3Jr-'
        ';i`lz2Y5n5`pT^th1Gv?PLTuKb#?brkKkgwuz+tT>g;EC6d<W(R9CnDN_0UI6m5%4+Oqqa}Hqs9-<+&-'
        '?CN~6wN4NzxPZ@eMopt2E{}e8Y$KZ$g9=9Et*NHmu6z}LAfb7bL(|^GozDx@LN?b-'
        'rab1)onKv?tnkW_d_4R5`*$*u!@TEg9!u*4FaL6k*VpppLqnX%R%wxfwygfxdA=t#njKAu~IbFN4`HMHp>`DR*pe'
        'E#Pp^pM+QrQAq`Pkl+>y}A_j4F5)bv`yup=CxbJJ>qb@H0o2%)u0@VIjL%9ScOljX?xCU*auG|*0s{Szf>D|k82K'
        'sT0p9}3K_`KZ}%lWtkf4su{?*NNk_CC6V_`+)(Y?D~W1JO(qX+n2{a?`UP-~8~G?_Rym-oASM_4n_-efjo$kU3-'
        '>CK-=0PE23j@2SJn4WnSQllYNrWo1{u`%+Q5>HM>+LJ9<jo@rR(QtD66-'
        'bKY=^L2+JP*P%!owR%T#+i}y2Hiw`@kaK*TsKo)3A>1563)Arf{lI!H6V=y<CVU1*38}+6FBInp^6ZCK5F2K_puT'
        'j$CvKwo0mZ?EXVfS%vC4*3qM~=3*&w>TP;8@a@?9ecOqBYewhG4_I{|~qn23!3n}uEM$3_g%Mr$lITDIox{8p`vc'
        '=(BT5d(+BY|Ple*x3n3&TVm1kASYS7kF%rvj?5S8)ToGOOQ<6k*kgob`;{(FQY6noN96GDEXsBIJr<8&G3ldBEnu'
        '1pV>8Tc3RHetoki9%0!{7&3GITbOt(yXWd(`@#y+x&70zXIb&j2h@I_KA#_@@A2MCmf;mO{3`c3hgX8hlCh>ve#d'
        '<wk>2-AD*d`QG6xM7qyGba3?}I'
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
