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

SOURCE_SHA256 = "f590cfd049cbf6a1a3cb7ef1728d4e82e4544fa86f8c9a62cf0ad6ddec6cfa4c"
_RUNTIME_PREFIX = "_portable_underwriter_f590cfd049cb"
# fmt: off
_EMBEDDED_SOURCES = {
    '_portable_underwriter_f590cfd049cb.reporting._underwriter_styles': (
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
    '_portable_underwriter_f590cfd049cb.reporting.evidence': (
        'c-qxHYj5L7a^L+cxC#)EiAI@OAcsI%EiyanO^lmY#<O=p41u5}%4Sv~btE-'
        '2KF9w(RsF2$2T6Np_mIP3$6|L^S65e8RaaM6&t|jN)pc96chBnb)VIYxDL+-'
        ')vfh+QcRC*TRk=<2Ch0a!TW+6ii@r$O^4PR}Ro^Vm&R%YdV_&vi0!`~~U-V_NtNI?Q9-'
        '3{rPYS5GD>i+XR5gI<Z_BPss>9*b7uWkT`BdyrWzt;#PYG52^$dy?^aT4$ntFc+f7RXTP<Bb(w1;9}{k7cA;o}i8'
        ')3`3qX0zGZ*{*F4NuKXc{i!YU96A!Pin?z4qOSnZ*_rxwDEeDjf79&u&^?weitCMT^rko-VT|+Svi!>l`;fJf0-'
        'Jr&b%2nnX+F(i?ByP?gE8z2-J(7nu8~5j_C?*7H$dlk@}VsDvPyq<wBS_7@7w0f-A5=eRp6tp^ST0}BMhk3H!uP-'
        'F6ixemlQy)qyBU(>MeYL{~xz!XXlD!m*t_T`)ZS{G_<R;1pYUxfl{B!eA7Vdx~%)zTz)^|To(I$TOP|=fY2qyW^+'
        '26aPqhLzS%&Xa<>DL>hkWkIOZ__(21b4fe6^`ntj#e#m!Ay-T(k7TAy9ms_*wr|HrD{dRTU+7Wm4buz|KWO<Q5-'
        'w`I{EilZ+21k4Sjg8>0n)utjt0%DcWCMT`kBK5ESjJ({md3R`<{#GD+{slh0Zo2LgIP-'
        '_1t2%<~tZhC+$0zJEnSCy+o7+AIZl0~rE@5@GW!J$VFM&BvteK)UHY)aeUK`k^Yie5iP}SApbjWeR0!exG_wX+aS'
        '+GBCv^ljf%gsToR%mke=H-9oZ(d%!&Ch@O>HO8l{NF!Z{FuLgck%Y)CA9nJ7vG8+mp^~_>E)~Q{N;aLT-'
        'wqXe|XMHU!A{xjo-ih=ll=PWf9ao|M2qF$BTDw^AGR-OiIr_od4&~7az`lWOZJ>d;Rm9w^;3-&-'
        'gOApM62f+~I$O1erYyumRTJmpd^7Elz;nEW{$Fn|=AI-'
        '1}`pyd%3BtgiV9tMVG0wFbI>I13OefA`_X^A90vr5?4iL29KjYGs4eN`vTTgVajLs#|KMs>VEvkSp3hkn7dYAO3X'
        'Ci1x>ew?86pfo%P2S$Ach3aIyCX%UeW0V(0)^>CJ*<*(2GbpHCc=sw!lj~6fh{q4KUj~B1VI+A^83dtW*P@}xx?u'
        ';x2O<!GC`>MawpKgl|1o#dWFcsk4Nh4`wkA?wCwK#SlS>}M%S$=u`=H**JCp2jB-yzjAtbV4bpdmFhpo*|aP?gTz'
        '``7P2MkkQ;#uU_uk{VG|Bg$$(VFi_z8P=<p?_d6K@%rN9#W^3${Y=^pg&=b>I}}x|DB9s~LPhbrL?sA>@4DFoz`e'
        'Zq@ALeJmzN&Ym^#Ep??1fz;r00&P%3{u{~+Z4Y}ZQL>kiiUi^X$j(*g6g$e7FI+r{&Tv$L~*Gv-'
        '$+ds=;LPi1x{K7oye!+6yIzF+#K82PR(4&@STp~a8T@+bTyzQYJk#eNBf(*o?Jwk__&=OeBLInt%J&91t>os)`dQ'
        'B#h`1F(ac26Gfz_ow5&lx-J_g=~bpUbGcx^!!v;Jy2(hRUH^kEvNu9AWF3E-<11(-'
        'hl~Xi;isrHufP$>vCxoza=WAC;WYVYHR4fLjwZaZ^61o2FQ=i=dxWUyM0qg$jxpC^GUHx{x^9GOBQAg{~4VYZG`>'
        '=M3=L&FRn`<J`hLz6w+J#1jsNT_^;?5{=IIRz3OhXpI61Pkdo}@y7*L7d)#%%4mYNV(B{cpFDcdv(@c(H47HymTU'
        '?kPHBBUXVt*9i(nHzbHd{-'
        'mt~s@va$1+fDtuG+1=f|Fb;1B$CZ0{oM&;R{dVLuClGUMx=O}7#fYj%L(wq!x@sRPVZObxoH!`AF&Q^eKlunYl>r'
        'Tu)t?6*F1t=Bd((s^LOKfrV0;kL^S2`-'
        '+Ov;F&twDhJ!oXJPED4vPDSPBnBbzBs+FFP~O%D(+Fj*Wg_|JL@(T}i6%YT7g+A;}j-'
        '5xhDH*K}ei!bGXutlAM`|NaE?uygC&!v95TcMoJ9yPnlY8kDkdlI(O(Bz2<`L-'
        '%<Y7nH=rW*zQwOD6V<m!DY=YkEW#<sdXDFx@~9aJd!HEwE`iNDcxRntw|oQ`Cvq7Nr`1iL%myW&vo@9gjXa|!yE>'
        'l`iZ#Y|M)v25ky@Cql<a#R$d<BAi>McS%~Wn7L2=9(%LhrfLRIEBxtMsffK*8LQSR;Mj}midh(--'
        '+mGRV%2#`D_MeO?f3#?H$XOdKGJLUuvx5uH5get^5!6PvOAKjg+(nnj56s(th74E0Y_YvOG(keIMd1p<j!tD}kat'
        '@DSK*Aco8BdD}MaWHagmT6CSVq#V4ZZ7T_~43g>09~AUVh?Pr_+S!_(jg59&wL;o+4yRqO=gPDYh@<JDysTHwn4q'
        'Gd9dc+0jvVuBHhU#;O^SLeWKsh(m3UZ>_UEd<)wFm9=JEElp-yD<AtSULtj`Jvk2_H@s#N`}7TqqXn?9+!s_w*!P'
        'h~%IJ>JZnTXf@!8BJ#jrNgP~lj|}mKqiPbU{lMP6MDHrpF11q3|L+2N06~4Xj)5<ebx0>oGQ3D?jPg=ZCb(-'
        'b4!e>R`gkb^?T@2g0&@pZJAOaj2@x|&Llu{Sw;V0YpMY(U<09~5QBWR&K5us*y7tIDy$Xl{C9f@jV*wnA|FyiF&z'
        '@X@SmyYPp+yncUK#sS8%2^Y&qA65Qumeh}`IOvuv353f5w94$RE1^tcv^*B74D<AP!pfX}K*>0qkKpS+4fXSJb%K'
        'vMpzFiD66k7kw-ka4DIOVIE&aqdtN8|^DXYfiNsta~)QO|u+06zHU(irrJ1AIkQo4D-'
        'BXpezf{MramXfgl0kYS2_y@<92Cq$(I5W>8n^Dvs8f52}xtq6FFeC$%svqob_f!nSAwQhmS2J@aW_wBob$N~p2Pb'
        '3Uj7mg7_@|2P>qjkV>vf%Tam_h<@=UXA)ut16kvKa@<cwkEc-(_p__2=-'
        'g%*_*sIEwSzCS7(6K(wtLEje1m_PH0IQSUh&^EjW19^O<E_5X;QgHP8kyNnMMUV`slx0@!cP@a@?%0AcKzI(Fg86'
        '^?dNR`yR~W)hmO+_T@5mXak-N9s4CrKFJ^+=`6T$_2FKSA;;~gQbo5Nx*4?o;K0nY%BeRG$d0-'
        '+mg?=Ir*p{xT{l@i#-CaWZ0MX=-'
        'o{1lx0+ktFw7KZ6vH9h@vg0*d+fLWj&3j6r6F{Ek&i0QN#k~rd&)QDnmuJb_~ux;a6#g>}GWni0Lp|Ai1{~ebYWD'
        'sUwg^UYGE%p0ADhRKudksTHZJyjZT6OvzXeE7nS%?p$>&Ef{%Tb@{P5$^)NNB0DD?W<m$>S`2SinS&2f(W@Z_IRk'
        '#%?eTN~40uD57-=a%?(N@5bPv2$*Bcm#RC`guLxhhcMbJPaa!A!8s>PkxMMzY6;!43`YQ}@alnTd)N->QSmEk^C-'
        'ZBdaHwB5veGg7Z{#h9)Tn~z?l-PsiF28a(u1sWdk%2LCFajwcCtX3r;8DRC0qM(5&RJ%Sdm@qX-'
        '55e1@1ptWeZ)n91t<y(Xo{i_MhZB-h%HtE+JdZN2EZGzlKyhcS~O2Gz9VLAR4#KP#BLg-'
        '$k((c)#lm_y+W_r+;KLq(u~{g(sCm0k)**D;u!J)C1}MVTZ_zq(8OZ%5VP&su^Om1K_xt~TiRnipvnyx?S0>-'
        'L2*nG#iprfl?fZrZp+0-'
        '>vBneR63rp>ICPg;}L;;&~ARTFIR0(#;O?wO$039WQ)iEjU&}}JP;8a?{di3M7r1hrYIbx_hncn*vaU3%*9bT!DG'
        '=&_v_64G|P-'
        'T4>mztd3Z0olDht;uKLrqoIR*R>#jqUxw0*uB^V(ffT6eIbU@BibWzm@j2sUkq_^%xfm<ds5R{0{;(wZ|PCICT#)'
        'GVtsa19?TNx4u$X$iUFn7Xc){Js;2>WJpwVWqcvTtb)$v<JS+CHis`OF5tUc+C<J1bqcWqB0NUt$G|78XnN`iVVd'
        'E`g+Od0^%ix=`;J)hOj=Nr#*;aF!B5fVS97U`oJ#mporQ&lW{}mu8QdQ47~FbKTU>YK+x^iI_2p5ENk?ws(yXpFl'
        '%=i;sfC=Q%(UrgVahLL_1}97WYPeX+-b8a&)0gj;l{gH{Eqqwcn7-'
        '4Q1Lq#(d3<sU*Y*2l$P%eLv#eW~U#14V<8AuG`lr=Ey0kmDte3Nj2?gAYx$;uuO##sDPa^zgpTm?DS@u9VN&N{T3'
        'A<RDfnXJm`2==Gk42wycjhhu+dcg;w?Cy>ECFIqVy`AR4Z(a+h4M!Y{yIGLPB+;`qljo)+`QFh$*G~pnHZ92T;l$'
        'q<cSq`Jc&r76<MS4I9K|A!1Onxcv9441QIG#|8xcUdF5JGe8$bt2xe4m;$J~V=fQUcBh2so`K_<$Vcp~j?viLuq1'
        'RTd@=?rd<Uy%QZgQ*_s`?$sZ-jaKGeQKQqh`Haqt5E?im-'
        '>ggo(QvQcG%3Ozihgq&pj}|(oxM<z>%SQZ^+$(>`ZEEg2G7kf+D5a;%>T%Vpu(W_T&OUd>9kNlB{8acvq$}_LaPl'
        'Fj1K++*v`JpY$vQ#1c6`;@kgi@;Hto^Y>Ou#stn3bIt2E23C1flyW|Tx0c7tMakWHzwJw2ODeA3*gkHF0fFW2OSO'
        'ub)g;Oe3oqo3p8baZLx}e;}y{ka|%kz+9(^bSEl4mVVyRB;JLlTpui7hmXniNx=nZUNKiK)jZdyLM(KVahuf2@7Z'
        'RWy2v9W0w_1K8NT?(#6RW4x5aR*r{%c^t=SZyn1b5)A{Ogv})T%*K`+Mej?{VkD(uw)*0LQ-'
        'k4$7aFK4WWqsx%b5MJ45BD!-rF}g#bG?%#pK{hgWZ*XBQ;9I>%b2vF!ONToa$|-'
        'C%|C>X+vXw%@N%d(3xBdxx984keaeda7@Y~Co<iJUvm1B0xsk2cP<zRAmW~o5&?f{Jkan%QKczKDvwGD-'
        '3crx#3^Cc5#IE|`<Drb4-k-'
        'grOG2z(H9xFwVSR$7b1W7Dv<Ae)YPu@6m#da5{oON0t4K*Qc5l4){a>v#`^U?&Kysw8stj-8^S<N|BA6mz&-'
        '38kui&C$Riv&DWi=ePGG`3EAn^5#cAixH~cBvst|h_@FfiQc(Adt+Kq$lvCb^zt_yTA(a5Ju%jmrTIO}x~T`pv7y'
        'EcIC-K&)kLq3Q|_Znt}Q4>N=xt+?8c{D%0%b8E<Pl1c~M}zQ1ul#oBQ~G2ey`I&Ci{5=UbG6fuJ{Fu8_-'
        '<H^NWx*JBg%)E&3e6a8No%Ku-tlG)%YKglIj+wu0NSwG%_ssOWUMh0%F%<uhG!96UQK87sc!rZ43;KU;T=iEVMWb'
        '#~3Y9i`u|2<A^}=X2uxI6CH;GK!-'
        'esK7!n?RNSUYHO~)NIqG;{m^ktpz$Fl*>|)LnZy_ulimGP1&pWwC>a118j3{xD44T!)MbU|9C>`Il2;Mo%$Q`ywJ'
        '6Ra;8y<@87i@DSkzJI3Nd<1%qHl7Mks!^4?JxI%uo5KdYkCO-'
        '+gEj;LdSsBUruEz@VY|a>uiA>ErH+%>5ze5*DA&bwms5;aU~t6Y9O+;@}0(7tEr{5x5mvqQ^m?+U8(`t@!^`3%aQ'
        'BunR|5)+NuFN74;aPg8tLic;K$Qz+yLd|K|2)k6<Y#OtDt-eKK&@e=wBf9zj2Fp;q+Cp@HA`13({OTMDRQoU1zmX'
        '5MoR;h-<v0=V%Mjs&p-'
        'bsPy;L+nVghTNJuHPoh0g861;TTGQG%50VZP>Rd`1Vu0uwP=&RxhcWWZf7i9+;r<IFnMoCjVrcWl*5hf#6+RGL!p'
        'Kr1Ee<WVL-AlV@I{lX4`T+kpdzg%iZdy0ydgfZp$%!FCi(xjawTj(4jzd_ehC8ME~oy*!)uV%s=H25C$avVr7aZP'
        '$vHJB3{XH_~YVU19xyHM88IF?9mul+&t140BKOPpwE&c{I}TLJ?n~HX_NJ6sq@>!zxoEhJ=zb9XUOI^aa3Aa2in`'
        'EWus@679ktv!Q({>)HKx{ik<m&xod<Io5X_5#s<6Ku^V$4mos=qPw?Q4$)SdT5*V{{Epg&-'
        'Bi%^XLuA<I#%L^fS)F#!e(;k*THP+i#&j{M&lZaM;4oLvvu-Eey4BgFa<{s<qbdiWevqE^CztyvzCxGxDmO-In#>'
        'c6LwE#Xa8-Rg^+J1hWVtw)*t39$KoaF7gjpR81hC8a4cmFxwMc<u-'
        'PF5avalfhGS9!_x^*mKB$C0qH_LJIiUUY_bq2b0?9RvE^MzGNIBFHirQ_0obZJ~MYzuKLs#e=VF<vV7taGq$VhLu'
        'V{kt({Pyo6@3xpVQvN#Pm&bTG3(VluC1%~5Ke^89aIK}%W_Rd~Jp;zrnCBs=SP?o2J2;8l`UZ69iho!MCzogr?(L'
        '3+K!19u*d@Vp%{1T8N*2*K=fX~!S{>owO0cUE!k?LPOVe-@SqMB9j-#9Quq<Ylhs-qpfZIBcA=<w)SbQ)I!OaFJd'
        ')nH>J1Imr&Goe?82W>zn3&V2Qt17Jwal%>FI$RQVH{VV*3ABC=daL@+jY(qT>2nzaOxa47Iou>a&g2$c{xDYuG81'
        '=U3v@aPDkN0tQ^N|VwX8th%YSK5fs81yww6NV>iHVQy#DoKJ(02rqwF<aNd^Y3&6%NgJ;++#<~VMAOh~*jAIaS@A'
        '58o)fk~NXfvZEFa!Tw@j1~7y=xlt5o(vcduvW+YiFZKmn18Oyytnze{M1<w-gRJLixdN(bxp-|hS#-'
        '%4s&rf%A5BFxrbzkdo+fRwFJZj=xF-zAdtoFK!&|Q5o~v0y!3kbKFT(Z+{J)<Ob6VfP9!-'
        'BAR`CRV|1k;t@(jik8Y*nSS>`NRZ#Xh0?LI4L%EmbjUl(1Imn<j2UjVz>8E-'
        'F#+I?O*uou)z(MH>^!5Tw6tkEGzMJ%poI@!A#yYhJE_~DJxb-dHqCC*6QC!uD?5)YhB@iD-'
        'd}(EU)8BS9;tw!!Li~y^;!MkC$IRHG7sWbg%0po3nYLAZvXLcyd&g@Cuj@)EXCK%V_O&K~4PQ3pv9~^gk<PeNH;|'
        'rnMELkOb@XIdgKxm$8|Y9wdojL85RzYS%ZMQ-CQ1?j=!4^+=W(h$Lycl`zCMVBs+)=vtADUR@z{n_LZqS{6lZj}6'
        'Gkcl&L0v?xfk+iM-~V-0f(IQVIz-'
        'u#^W9(?xYv)Bb5N!w6=?on2VBXZkFC(cN*Y91FE&nXF{|=pGM4b{<UUL<fJVab*YFSm!Ra%%TONUD}(gfxdMZnu7'
        'PCg9*lB!Kjk8fhv!$h3WGUj(rDUYGPtn070bZ8^n(r;pv*C$N9qO{W;#jKFLm>|R)lj1rO0Q18EjOmM^f$<R*mj&'
        'm~^U^Cf#ou25EGdtsBVMEIk`stGU2~L4mSTEGAHu&XZk$V2gWMtZ$LKNP3B=e-'
        '(D&6Ryb!4%kwGki&3A*muHyqbrTej=Nf88#x_POI&^2>8mVPb#OVnE409(nHHa9X5tZ^?{boxr%0M=Hy^cbjT&1='
        'WWX-7+K4-'
        '(S2SbyTAc$*f67y~A1qY;w)j*gGK_$PKX<c;_coMUnP6@GgarKw2bTe`UmIh+z0y{ZW?HnWUrfPG{Ts#0^#<YD80'
        'hd3yXXJGseWL<f4futfT()fseS;EcdFlgy<`2(JJ;_Xaj-vi3cl{a{)i|ak)or9rDbhpGRSKb0mYw2FH~efQy?^h'
        '+O*KuhNn;DufMA@;hqY04`~X!msgw{D-'
        '<;vvCA_5hcos+aIB{x4GKl~_b>@xCwOphCu0v#Nn&HVml+C;oqr95F5SA;6+#jz&qJL?&l-'
        'l^FGFz+jvpa#Pc!NdVg_Qw|1vCBZ0E2K5)Vp=<u8_^?uDI24!XgnL|2&RqrK_Gf)8^RI*(fzd!k07kVoQ=NBlNC`'
        'uI+pz>+6}3^Lti))t?kvWam`a~V16IJsP-'
        '+!QbFvHW3{g|46k^d59K4vQZ!*EAv6|3`0GU=$#^pO0S(3ZjRst%H67q%twMq&vZh4hm1~T@xp2?`p<0zCn7cMd('
        '&^zA2xosED4xOu7Qqhr5~eM#$>Fw`Wj$*kfCtJG1SXvaBNx3IkT68QeRh)aOdK2m(`6h(gLZRTM?Pk&YHi<5=xGu'
        'eO<h>~T_STXhAfbd|3;=y{{NWT}(bEhhody?&yv9cH0>QM|>kQq8~Ljh6TU3-L-'
        '`5x0x31b^Oc+x$?Nr%MK|z%U2p^dQ5b`!Dn4&P1Qhg*r0mjNFFNh=gy!aQj-'
        '<UAi$T170MjhO1>XHjbkUmssEe$>dx&fgUDYSwWih%^sS{?6Ci`cn$&(WcLo=>PqDWjYWAhX>U@Cq@8}i@)?7a0l'
        'x+-evgkZCI86#&7L?u4c83ty2Q61>g2iLhZpX(9D>Ix^mVb>SI6`@z7hpkX3kn06uv0Qh$Br!;pv-GxCH!CC*EpY'
        '%!L+*)hja3L_or06&@?%-av9O3%qY7BoZv+E4dj^dqVNgTG{&wIIi`-'
        '0v+XPI?8+4PzvXNHc!;YJN6N5Mlc=a8VpF*NaUtE)rm^hNn4vC89V)=D&KfG&k9bQ&|v(cWBzl<io1Eh>&Us2*Nc'
        '@t%0a1*tmy+Zzh(&Dxsf_m2%N;w=u77~Ci*DhBaUbIU8!NjpKxsBgr$yoZsN}@GQSN$Y8{wwH(-'
        '^zaYEWH6uIgi(}e1bvC)FgD$2}IgzV!y-@_9W+5^Za&&a_Mf2NW@Q5ic#3a$Z{jB!K2+vx?Gxz>SCSz{*zHg&-'
        '~bd*fOnZ+i<$w&7T`@+YOWzrZhH`qRcO@V_rJNHTX2xv1s0AX*$D+#kBn|eMpGFSq7#PV=?(0c<l{QfC@`}EPQsl'
        '09j(P{xMjOMSVK_S!F)RF{xEQ{Jr?-'
        'E=&fv>I?8o?<iv=wLCo*|5*%!h%y5o0dVr$@8O77Qfoz&b`l`+0u;zJA)_2j9-qQ$y-'
        'ez^?M~YvG}=PDyw1?CNv?FI)iTy0;l!N^lalzu{+={5|Umzl+})y%x?=8)LJ%Y-'
        'Eo;Y#s!M#%s*N*NSQDQgJmyo^Z(6Dv?ZCf_JZ(GY`GKwLjL`Tn^j4nud2QR~q(}f7=_xmyJ>N9&4rRz}-'
        '>^R1iL)R_r2*ce%ne9&V8ht~h$(=Wnv)=I#Zox!A=ZbhR4Il_l$XK5n(|(mA|08Bb^O0EXRV_$upxfs*UsQz#1pe'
        'ic~|1orwNVB@!w!?C#t^rQ01{0?Rvc*v$OIq4+A&!#<Nq$jdBXFltN(OgI_0P+iHxN9(->gkI{R=GhM-es=n{Vge'
        '(>tBSUOph4%25Jsd^++`cf}-1$HPG%x3mI_%YbFxPV+-T2V3%w41sU0iMe(*Qw&@=-'
        'vyTph{Y<;9XWamU$w9_L>&*fp?=j<s4<DDq1&92tT!mXqA=X#?*hsEa9o>qH)j2+Di_f{*mDiD|eM1hbH+AMndcz'
        '3%OAR#iOUx9wV7gFE5qioYM<5)l2*?mk7ZC6^1gfoI`%ao#IxKE5T3KW~VHcKo3pekZQ~ZsAc=z~2r~v{j&7C7-'
        'QIwqwH^LOEhwJL*)SSA+e}rTvK(-}K1r<G=GQ_wNgjibw-SNrn^pkCbU^WAb65_HZ2JWL;XWX(7-'
        '+WoF7D9MF{bXl8zp96~Y4E;{(goD2t}ar-6e>_ZHD>mlOj5Q?lD|uK2Uu&}FV!&sQHT-qQUJ*TENje@R9emTaUty'
        'gOkPp(V}O<UFjFP7c@}`G{7DEY3$(@WvpKzw7i^BcxSO^Jf<S$9BZ62ki;sjf0O~^-'
        '2Gf=zIh)MlGYSRc=L0~HTljEvk-'
        'f;1E%N*?r(%!BVxG?hpPLQ?_c^r=b+vWKCK9Su2~QeuF>Vt1Vd=kTA}5hCK9|KN2fYGC6VO{Tfe<3-vocy-'
        '#Qm>zk|!k4NLHpHqvKbkiZWo8W?w%oXOsgbx4HQJ4*rSJmATtx-'
        'N6{5ZXngnBj10XY?52G9t1P~ICC_Lh(GlAxIm)E=6R1ifvs#3_S&>qslo9O!v|U6m-'
        '5bgd?)ng&QNA}0fP^0EHONLE@(KR&NvmFc(!<cok-FbH_1yp?~5J`!T|dQM)`<PDnAc<08_U&%L#N1(EyTgIwzXh'
        'h!NY%Ow8P%@h%tBF6RDkav#hc^1!-'
        '~*LQ>6(8R&^6k{2Cv@7^tF8(jb{NZ=+>|0GM=h?nsb(y_%rSFU3lesD$=TMd~wQ|>(7!T^eV|e~A?>?lTx;RX@aL'
        'QaI+EV&<nlGKn)1l52qs!zR|9%aBx^G7fzL^ngWqpqJHM=Kd-'
        '=*)GdR+}B<FegqapAR}Wk#*E)%D3f1gj2IK9zXzNpM)Os_d;tUc@gdvLX|VrfnL4aN-uFCtG6OP!Yqysf(o~l*@u'
        'ZD<i)yI(tz#zr6VG^ZbXGm*<xg(I7B1dBIKsJML8X7?q%ni_<|ntMrY6NV;<Z8_zVhRhf$>oXD)m97T7>velW+$H'
        'CkB9Jc*V4-^e-cN0vCWR&?t%8_=FIPzU_sP?YvB@kpeM_zuLxwp5l%*nr@TPx~v2fJL4-a<*wsMIp-'
        'x8lp;jNTfT%&`ix%$$HT7P5H5dleb7ThCp3yp+9Rv}xK>`jle8Nfi#46a=0{ib2ykiM($VhMa|!nhbjZpYr!gPLc'
        'IK8qQe}9v)2?M47~<4DxS5MR`TJ-~={HxwRTm4bF9v^2-q<DQGdn*)})rY}d-?gYRZbyc<z<9Uf8I@~-'
        '>@!nha3H^t#l(86nASHQXJu~ZRHl|Ni*iFj-'
        'q(1wQ`_lzcHP@=Nf6kWM%_S;m5We3rzQ5`#fB_{XsUzMS!sx@KvDme5$gdKZP*M<NGYKprrWc18?Cq%7MwmL5b2E'
        'cu%ioQ#}U5tib8{kPy{(8sIFN9$H*1IRbnZRg*_aEN<@cR5s{{G#c&p%uaPnQ`Jlfn1{wBPNb>NbF1nJS^i`^u)z'
        'V{ld;Mkr9CGbXv`oi1<KRC<<}T%6*+<OOXlaDRgDYqG7S->J`!ZXd-'
        '?kOANuK<o>Zz3g*f%eE7pP)K8&Gy#$a$I#|#-'
        '{!twgnpVrKSxCI8S|D=2viE|&@&5_K25c!!F3)R`F$!cruG~v$KMoE$iB&*zk>}@(Fy^+FYb~cYMd1#ZpXmhs*gE'
        'KgLTzvk;YN#`zix_uy9%{XN(eHa679vZj>Wf-ZcSEt6%#A)MY|ZQ^tvUFxuYa@Yj|}d#C>(D-'
        '2JB+(dPDJ^714gP4#JS66lO-'
        '{LD?Vz`MomSBs44T=&)0RxI?uqp?g`iU#qB=hr0aB{c8%xPq(5q2CgkJ^B*EFy!Zc%$@-'
        '+7zuJFA`Lrhj;p83zGD}B~%caFRD-;@dvxjuIbgw_b-3AczyBl;(YA-'
        '6+?r7^4=#2ubs0eI6R26!x@0@s8F5`<}1TC@;lz{C5{;R(S4>U3qR@;U9QINRDcC$cp3~FgdXj|!S~CpaY?x;^ny'
        'HNtMFUk{*&zEdGDMS56+~TPE>yL4ak4^LGTLXA#l=}4;$6~BGH*@on`|POCP>!fyEj|bB41+&B^JVW~|wGk|#WNN'
        'w}v(v@}@B3hW}Gzw82?1}+dPpNa4K;Z7A1B1u2Y=#kYxbA4<z;~!M>X1D7~oz0mV{Hj1(^71v^NEfm$=|X;({hBv'
        'bl>16KsBnWMXs)ug$}zCdsl}KXSXf;and;euSxe*?gKjQ4p{|!pAvOKpKcA3+oBe_;i+WofM1m4c1W7GQ0~|Pz8T'
        'PTKYZ>_}n~J~1p}0D4o`hF$b_|+8-)Ddk-m5{MSE3+GWJlNU5hIG-11&v-K&2`8lImoMV@$Tp($7!7#1SPqzp$-'
        'os11SXA5yUg_6R7$Lj<OOzJpWft|;4ptS0+$mYL}u)A>FMk6`7P&{bK3Y3^h?Uw&5M0imes24{*s-'
        '5|D)DnMs2gK_vdX8Q!Q6YVqY`DGD1qp!n^DdwSoDsnGq*ER><19#bD9G^SP%yZqFlE+h9&ly^xCUSaw&AK54;>=G'
        'l!KBJ=6irM=yDcq;)LO{QJpYu}=(~x^Iog{qh9hM00yu+oIdgLQhngXU!dEo{Wb)ciqLbfhDxu>2g%no6YiJ&mv|'
        'iUaak0*+Pw`=zyeH7T&5UvN@xvJ1cnbzNPK-I}2^Wlv33p=Lm8hPRRynmM`5C8-'
        'm17ll>r$Yz$9Qm!hi1T0JLIH|AQ^Yjx|@&SV@c!Dw{}B9POAf<7fd(XAC^Y;8%~-'
        'Fj2Q~9h8VC>{#f2=&7dz<3OFi@?Y#wxzIPjoDKFqFqL^3<yCWs>m_u7KyTjyL*vGsg#adbv&3i(F>3Q8$U(PjSbN'
        '+k&=H=Uu7q2Fz&7Jh7s(q=*k6;a4jE>+%!%p76e)rK$zB=+agbiewhBEgg$7l-wj$}M`f;-'
        '0&9h2R*c5CY=fZOVrGUX>Lz;c*<H!}}Wh~MAa-;<LcqvjBvNn-b*^@OslIrYhZe0=jd>5H3gfNbvF?-'
        '<$q0MXwsV>HFvPfIJuA-'
        'BbYL$DFCI`=?#+V^~4%_EUz4ZClA%*;5uJ3bTzSd7K`fB8=4QDb4@<I#ZyLT`to$rG53l4pY-oksKWf;7`b^WTo3'
        'iVMY_U7d@_PO=5Qo{>E|lekEqARz^v{ex4dE^*s56Z-'
        'Al;n~9@(vg=U%vN#2TM^k)W{WQgxz+}fI@(a{ztIe6($g#iV{~^e8?jt(SFZb9HqMxfT9ZXy7g~oek<hR{d67gm8'
        '@@oojqJh2FXmx6a@!*Cq($Tk#N$}R9R8BiN6H8Mj55rRL(80<MTRt1Bt~qUVl1|i1TT7dTa>Rh0AKOM+5Z8?tRSW'
    ),
    '_portable_underwriter_f590cfd049cb.reporting._underwriter_movement': (
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
    '_portable_underwriter_f590cfd049cb.reporting._underwriter_html': (
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
    '_portable_underwriter_f590cfd049cb.reporting._core': (
        'c-rlK+j8SZvfw+v0>gbF+!XsV?U`wvrei!k?KzI)+qT>1oCt*i1CgKvHwn-HsFtkJ{MNqjmu+6^l63(<$sWf>>^d'
        'AR0;tN$%F4>j%F4=`PN!ej>tcIS6^E|Lw@H3^*%X&~S0qibubZx{E|X1D?~;7m6#4r0q+Qibu}=2zPr2&Kx@s5G>'
        '2xxoLRq#sbcd$MvZUN$^(3#Vy348DWTM{f^6rYuuj=i#P*C!Vm9F%4zTZQ`^W^*Dr$bS#3NF3QyL`3H+qP(RN%P8'
        '-D7tc2=+8gAe=|2PxyT+G-<B7;#CHal?iSnB??10@=gC)PR{&8A;?-fdzfJNssrLHSKCjmB0{-'
        '8xH4*Sbez7gGL$xlNj}1b}HlMzFw%V+py?nL0SQy$aoHAG4ZmSV!3j7d!{-'
        'sUq>g!@xRGn^SHc8;WtSgJG%{N8%u_!OEx?-'
        'JOlvO*Yk9Md{?QEalZtHwKpQMjKsJJfIII1oVUw{7J+1H<c`Azori!a{3d7u65-'
        'IssOzWeseZ{B}DHwC``;oTRXzj>Q|{@-'
        '7I?|gi#V7^0+X*>1uCSP@5)orW4DnNNzz1?hp@x1pq6Wmwt2<bO<)fG1isNz|m1(ePTQ#_qazH4eA!S4O-'
        'zDSmud(NhthS~UbI!~r;aa{m)==a5;sj^kwcGL68n|gcLRo^!MnQx&3P`&M%<R4}RoNa*K-'
        '8sCe_6uOSrpa&NVZUB{58#Rx>hJb>Q?_;WLsfQRePHN;+LP?Nci;ZytG8cge|`JK=RbV)9*Fy2Z{MBjsdxs^=83{'
        '#ncNX2M8{5(=Zj}dAKSyz<b`=@yNX3Wolov<li$Dn`txtzfBEKnuQ}%d{yFbynqSfDrfBydObU2AeY4HWT_T8Yi&'
        '>V&g5U8INVQdem1`Uyg8TMnU$-'
        'FS<k!`qxh}FBYWA=@Y(e5(7fGQ?Chb)oD}U>ir{+&q^#SImta2YxcCgQu$BylCtcL9|LnXY%JKfibBl_LOyDfRAr'
        '{8~pA_)~aqRSVt!oTJChW)&sOeUnqk{_)6Wa0C+Y3gS74#){Qj2_Zcrpt8tE-zc?(8sHyO7f&@4sG`ls$AVF?VYS'
        'h|0Wj&Nb!1oSQTs5){{RQ%{c=B{&P_+-'
        '#3RM?L*?*Jt=9^@L$VzzXcfr+RvB_P#{@d)#VDr*QRcgy3$4jw~1A|vfCD?B%78vB;^WqE{ReodtDYEr%LCLB+Wo'
        '6LvdHG3VvXi9}5`S0#wk)qB&(D1;XtcY=23AxLCc<KAa|Hg~IXq3wpP%52&-'
        'Y<)$NrXFfk;QBR}je&8?b61q@c97rq50xwhnqbVa`n}YZOrW@c4UxLc;1e$|Yu3O?y3Na(>vx~a!0GE890j;;?3X'
        'GiqUbJ`%R5OVE@x?1fyR6FHVVA9n?KW%6p9>3$2XI|%z{&%*?=mRbW!bDPwwp9L`HU!iYNh~<G^!!^yr72#l59Uq'
        'O_6ewAagQLn72eB<m-e~A+!IK!R>&-'
        'k(sNyIw^L0kXH^gYD=;X!7r<%N4t%*=|Iv=8|5%JLUWz~w_B#PC=o|$8i2T}x7T1<Is7_|{&Fcb9nCg|`MV`5+_4'
        'Urs%jp#$pK1ySC=iY0u1GKnWGstb1+&yPsgZ=Xc>*+V)bGiD_zq;>TmgOm-'
        'nLd;wY5t5D-o{_CTcMwZiW8F~}Jhu2?a{k378Uwzu#b3slK-YU_o-'
        '4`pAK<07O|rmay6B!h&WpqIW7^1oXt6x8&kkYB)&LjK1<4EY5@F(gW)E+wy#hQR8V1+4bEhNQxr$dEealz>5rLh~'
        'Q#1Z<-'
        'ylN4JRkLOMeB_+xhy3iqW%&4t73Z@iiUibxs+?GBuR<CudF3M2`*i7#br+YKtc}3#*5_q8AB%oQgMGjx?6v+E&8e'
        'ofkLVgShOuWKKeKxe<ISSDLyu}&T*vP_n15FGe9vS!m^T<l<D6B)+akL}wCCXOWY56*$6Q%fYXQjtowhFVDQ`TiS'
        'S(JFrFDptQmp>yaig&Nkph)}XR6xCezSEIT8`Az<G&LgB!_9!j)B$dBLNj5I#8_a`KUwi}bguFTGgYab*U6}W?#0'
        '2f5t#=SnWWc+?ysvr204KG8dyT3+|XAo${F#(sH2YNhG|%RL};OwBA9;$-'
        'MKHC?$+3JaqGkycVtytgmCRqbhwV{u4o+u3|Dqa(L(p|t<ujYq@SUU{IKPffUn41O<T5}CCSsB8jIU;XMs6zKJh9'
        'N`m?aL^x;dPcSQ>wF)N`T+G2B%epBP!;--'
        'Ny^KD&Sw(yq7^6&4z{wmp&U=N#BQGMw42Xv}+SEuZeBvo0UtpUdTe!ciBp!h`t@|o8M<Tpd6^E@W9B-'
        '5a&KeibdSZwOkE0hJw&zI5X^0GpoE|(aiM+O{NqDyuCWd%s9CE*5Jj9?0a{*Z4^h1<z<DYjk@Y(d7K@eBZLiX8pS'
        'P`f+qw*}W-EEebfdgp+PlohNm+W>}Fsy&uMG-8`gTR3Q`6(`V-'
        'mPjjd=Vf)1T^FlP%kBu7*rFoMdYZUSK>k!w73|=IW}<D9Ne{Cqw1A+v5Fb#4RkvCW5gH3dv5iTY|0r%fbBGJ$<lM'
        'pP0fy@|xg(grYouD%W+1h2Mo;IO#o;yJp0|X%Ms$tSbxG4R)!l)z(8Y2`s~nUj+oGDm%M@SWZ-'
        'OrRH*Uqj$eaMgqH2NXaGL}4@GkFGS2QIF-qf^@R02%Pc7sdYf|_3B+wJW5CRcUU<v{7Wf~5k3VcN8+86}qhwC0#='
        'HLqY@jq_@RVLq8V^DD+U{gccsea^+(d<Qcr(-'
        'STR(at*KRz|n%+?f)dBb%qiJT~L7tmCwnWozYe>h(pb=@6Gkp&de$bR{I&9XuDvnJ_TAQ!3_}7*t8`02!!ifEBGW'
        'WD2xVNlwjxTr>40rG=t;pjoM?>+qoqhAg<-'
        '88IlLtvo8ulGCY?4)#ek=(4P_7SMlYyQu^~<;)bEj4<dkw@o`wABhU~PouLwXk%72tK?5+s-'
        'SV|qmZ2NODX6K);<IuF(dt^%8P26%3-98{F^3E65PdE{0PgKSqlY=^~{HzrfCmDis@;#I*22&oSeIpArF-'
        ')HZ+Jpgf%-'
        '^wE=hwR10;Knj<V!plAF;(VYduP(4S2;8vG>7UC6@05GwF2PVJiwfp5W_%hUa2fX7NG`9c8vb%C=v9H^bK9MvIkh'
        '&#&hFWEsM4MC4OcTq*_CyJNWn{CgRNiS)dHh1zzU&6+r85@>0`m{s2_-XoT#d<R<ep>SnCkACM)o{;y&R*r_S-QD'
        '+R!IvdXDnQ_2_BFGpIS*34aClA8z*DPqas^$v@KEo+~|yqD}@OfPUDuQbV&h1Cg8Qx@n&anry65Z?JAm?N_i!9&t'
        '&NnBLM_Dn*p<rdbB2M3znM{DNOlyO`dOps<PKfLsG)Et+fG+p=3fAPSm#3)CgU#I@73w+Aqaj>s4yMk*&!l%j2lk'
        '^@47M|Yk$o$cuXGqEk#&&}iX)J?c@G-PeZitFR(GX#bVY~L~atAzx7ftAE;_D>BKIP<WZSzX~-O5viI)-'
        '~5URsbo{<ke*{v*xk3t<jRLx<9;3Ep-SA${1&fAas-0yuY64L2`U`5nW;UU}8%RuZ+tIeKhlIp(EUrK?&c<#(-'
        'gc-<6vZM5@|f1!-8bUFcd2U?fCA!eSxPqoAnUMIQ~l%JSX|-'
        'v@|2TG_fQuV8SQPwI;wVIn#c%m4_NW9C!uqHNTCN1b^10xjfzgNw)Esuy7Lmu;2LM5UAj$K#Nd#;IaZt&%{D7&(D'
        '{1BG$d+<Ft291?T<4a+DqMu6xhM_E|@Tu5wT`T<WZzW^>%yD7!Zs@Qi4Zio}A7=H%uhP1ce7lpE>|5e^zVf@xK<#'
        '0cMCXIZFX$m%ywlu>1;>mH7YZ?nQ#O=zG>}+kaxpM-hO>e{Zv?pIyt4>T5qIUal$t8^0UC<9P<D5mgzofL76CW;F'
        'z-{Z5<u^3B3fvN37dNwYQ}5YlX1B{&)h~s?HIFF9c6rC@lSfDBGrhu-'
        '=1)xXXglcMf@dRi$$w()>$S?#5tsIR)Sb2z5=HW~R66^#7f8DdhQpTa^S>?CUt5wb;H5UBeFNLEr@fCpB~tB(F(H'
        't<BCps_<v0)L5vQc@BX``+NGUL#^GoHPYc+&c4N7X-'
        'uA6lM9n5IYch=)VRsI?b#0_7ExVa9ac2$5dN2QAU63U6fPE^@NVTW^%2s3SJT%tgl5eD<*191Iv+IB2H<Z9P8W(Q'
        '1xp}%LRrzg+PEqLfS!9=qi|DMrXw}4rODt?X>QA@O?BAqO9Q6I2drM8-y><lOB+-'
        '?8t`RO^THTNar%1|zw`mkpJtHTa0HwzbC56^8Wi7^zv00D)OaWnd8`M~X6>eM&gIl6xGiAM_tcw_qfZ1HR^uy-'
        '*{lcyf;Y@D8}Fsd%!TAW-Yxu9kNrzjfar~bCg4Pmi4fDGT3ABq`2S9z5hFU26js&{~}Z0mgDHgT(G*Ci;8ti^D2&'
        '=jk7cz#)-dfX%v6G?`k&7FWWHJK18Tm-'
        'q(UHe^Z%X)z4b}FgOm>yU4ZRn_GGB7G87!FJ(v>!=4VT{ZPc?Lc~Sh%5D{9Kpo8Hj-!wItaW4aQ4nfVloS&;k8-'
        '2zdo&91GK)ZCYam%3ueo`e#~DX&r#43xFW2ZQT~S@;o61$}ifitGCPN#mVzeQ>7Jv5>GW`$vE&~E?Uq$RA5Yb2Eb'
        'b~J7y79Fk679VFdc1qCl0X=OFY%gIQoh*+t%hI-|If`eVkyxZDLfNsT5^yN4Nx<(UdABnNq5-'
        '#7|FMp4d{v8j7p_!|j9_mnP047KN{e}hlW8R?Qg6;qP*FEW{{khvEN&133O3Qi!Mgz%XZJQd0s#}lCh5E0*joz5k'
        'g)CFOC=f#_P*gEzrn@`#)r$P8V@=%=fkfd&KmecLIXUDordk0?`5!)1?*VZtmTabrm)(3}21<-'
        'kMc0Q5uY*H|x`KeGCbi!HQ0eF&;7MBGWMoxGba=a}b4FllsNZ72}Ze?Hef2c;b<yBp;vrWD$x3?pObcp6q8CY0=d'
        '}x_nh6=Xzp>atZjz+W_JpF_9Kd>fFYD<*`IrUV;UyFJ#kVh_K2#ThYe%C^bYNAzsH%3BHAs*s7L{?GRz8|HtsKxJ'
        '04I-'
        '{!dT&}v0maU`s3ajX3^c1$1Y|`ZMBke5rm>vCJ)rmcNJ5XOI|787!wV3xpkiGXx(nmQcoXQ9=X2z<q#6`+o+ITQ('
        'Sl&*bEN#fEe`8CYhFOvxU(&oUee#|2r|941L@?bd>4n9;*Y}T$Wt;xT;6%=wi`O>BK<_xfG!g2M5yNS64K3m6RBe'
        'uL#QOED4vEY$s>GR(Ph0z-TH!e6o77-DeSh0cGDObSJXj$_eG>eAd9A_p?n&yM`FjVMz4lzS-Im@weN@PvjlSM@;'
        'kTQecbCL&7hf@)}c@0r$b(K<+f0Zmxy;_&<lV;a`{EMExXbQ<KUBh${-'
        '_ifZz14{k0<_viMhP0w}M%SPxsp@5(jB1v~3&Z+#4t^x5K5vCQT7V*3?$#FrM<Q?I00s4DWG8D9-'
        'WDDP*MRJiicK#iv$<){l$igzRheb4piiRmXp_S*-'
        'T<>Hi)8YOG5d6t`ASWuW^e=&D^^|1yXEC|==v_~Rc<kk9Px$dsO?4aYd<#j>+J4dr!=7+WgQJQ^N7p@Kv>h2g@m&'
        'w90<jYDNSa$6jCWP6MbCu#0kM!LvKsUg+%u@e+1@{b!s9${rP0yjIrGi_=B?(GG=Y?0=%=0M^VW=&CF86U2I-'
        'BmFJ;zPu{qtw^=TrXkO8t2?ji@YIp7cqX*nfS4b>zS~2!oD(j+&bO`4s<oMgP3&LA2*JZ`P8JVLq~#CLr89i+*C3'
        'K<b7(`%uAarm;BBcZBm*d5`qbYkTmM(an=~SJyb<E)H#36>WQ>aZdiGa85oHO;v0)3nv9wt6(q{3*M1_e^pqPMWZ'
        'TD42bW`?R5c`3DD)REs_K6b>V(o3)ae(gUfNp7;s&yNOCZeNPN&ihx!6{m%jX;?~-'
        '<vZ;J&U(+9e731|wvE@1gUOPpeYa4ZBJ;mc)HuEBu2u1j<vVPyN?|4uUH1~+{05b6n&^zVO90I7P#&{3vsL0PGQ>'
        '8d?6WX<7L(*`DS-7dKAn-aGh$$y0?DN77b)pj*)A7Ri}zR-jhK2W2>SaY~u7&0k+At>v8^2rO(N=E6lFy$KUO>&|'
        'E;!6W}BOz5h+34j?u<>A<Q2R%0=BAOY2%btjs+k?pC7appPoVvG`OQqXMkcnZ_R|ZPy+8f&QwLuwOkAYrIPqW5bp'
        'PY03IMZAM(hGWW(ey9!Qxu?`QrC4U#Yzh_>L6i5Mvr`{)K1mLgeQT;M4|0f0jHuw-Cu<sAdeEN~wluae$qMg~W{L'
        '2DS%u`^7wgCB+{YcRP!gD7>Hq1O0?*deR;}odD#MWOf2XP==E>xha&xj3QHQfk4pk(T{*=8a9ljjA)CfBCn07RN+'
        'tPb5)WFdL)FS)8{#^kFe*^=T~OajMnDL1%SnA!R2C3r~>N0C!5V+#n>NU<=R9$n<h{EDkg+euK1|+IRg@=Nq`j1q'
        '?6<&`ItQ{BqmhuiC=DFk7k3!Aag_U%gbl4{xGLzK79@x@lsGzOM>;K-f`MOy?wrNv7UQypTHpS<j-'
        '3ek&LTef$33S!ur5WBrLon;K4jL)rkUayU1JInVoT{z1cAlhBE-'
        'mG#l|?5ahGx=Rh?6`RrVYhw^flYZ6%Ydum~*U50geXw?vaaz|ph9X(0#H4f&gD7UkhG};!rj-KTmTQ}0_grU!MD^'
        'CzqaGXFXKdM;V>~Xez$~z279BR-{on~d-r%UxqEE+cK5@3zW3zfRM#_i=94;j~(Q%u})ipACHh!r-'
        '9aUL59=}dLy)b`0Cmpa{2R2|1Xb6?aj;r1j{P-AwgM(Teg|4|hCq^Rn{<&`>U#5>^!qbIc$!I1Oq3Co6`Ym5WAE-'
        'K0|p`hi-ruYbpc)(yMga29>#SeVI344?<h}~gzl>k2!%|h-'
        '#_RIm~5+R13y?Chx!XdO;p25OH5n}GqRB+)u*%q5_NwdG9C<xS5yvG1qT`Eeu;bhh`H#Q!RN;?|~8Ykcm{OD~t8W'
        ';@U3LE{#RR?CB&oRwM?f*Y?e^OYXqvxaM1IZvF>(Tze0=woEP6fR}&8rU>DzCTv<>GT=;j!{P5s{SK`|>(p-'
        'D(+>>1Rq}g)e6!5J|M-hsBIUx!BL#;b5H^&}bQ+0G-'
        '(N$6!o=>{DEx4~7H`RR=L4BLUb6_nXE8aFaNP?Z`NL{|YBJ0-'
        '!IFVDZe^UU__!KpK@x1n4<I^Az$x2jZyvJM@YUYM(yng_UlG9=gUPa2ekD?u%rK&I;?MxHrcUDTjp0+V)tt(~wN?'
        'W;=SA`utFoiEIhT(`E!ZIPUI$xg+XdxJ{!xNE+YsK3~!n<V*E4^d~09BYqKu!sH>Yl@XUpJXB9r8j)}h7HO^{E-'
        'D^P_@fg6GM@~1bEcO$CM)^>R2-rr#R89zv2PFRA^T7FPV^7YvFwOhH6Em)C)Gt!t*&-?^I_`FsN-hS01u3RiVgoS'
        '9r_0k`@oR*O&T%MM@^RhM~LLj35QC|<_;fEMx4L#Se;``a!eEud~(97q6hg^J|AeTgvCKquqRIi)xk>HKu_hyt6r'
        'g~C=4$z7<Wl6fAMr0n<|l}WY#iseo?jknPYH)66|p46af-BfT8BOjbvY{AV1Rs@eLNU6bSf)8%cSkzO65x?`P`Gi'
        'TdO>@Xw(Z{^=imk_3AZB^ZN<Q3_BnNW}=5+BOE#k74e}s;G2RCy5&?GeBg!g2B_RIIvBFlDL?yOANzXPB~0|igK5'
        '-3Ya(EiI?iIE1JBk8{;3@J)arG0@@?-#aDW1eT_}N`aow|E!3xpgl2!AkRwq<)D^kjUcPAcK)*&e<IBi-KP>Ha1l'
        '|6)`k<<3Sm50Gu8`v|js>s(^W8!OF^m*>;`D-jxLAv`<xr)0@_U@1e-'
        'JZfu(o{l`#8Xt2=K*A4`SNKw5Gb50CO{L2wqVm5-'
        'J!u_3C85izmHP;uD(P<u@qCXX0!K<K}wic*`KzWrO8ck>!_7K{@LlEozSm&hFb#Yhj{bLRG!{^i_$OVk->Bpq&-'
        '$HXhazS*}kuIVCYA_kzi9q}UZ(&kA6vtsmVDVmt`aey*0C57}gLV`XKj<j-'
        '1uyUyNp5{*6cs+#Z7+NxD8rHqae*m$kzq89Fnhl2zhY-Jg|>=czaq$7Pi2A+p&9OSC*2jyTS_0JH%by3w|{U|Byc'
        'mGMO4^K=>YDdGRBW)|ai0+h##-fpN0Y_39t^C#G5Khz8w%pHH=m^3*O`LX4G+U>RxL{~=514-'
        'Y%+^t20!+UO?ORZ2Ai!{4H!~ToL?n|FY)*kLjzfd9w)1)5XcimXFWqV%krM|WG2vptHO_GlC1A)QEnI$DIGzFWXD'
        'Uo@%7^^m4V~?SUY$qwGZ2#W{<)A;LV=?o4M)P0S9<P|LsaY!t(}ZSyW1Wd5$bLUH+S4${jp6h`NJS5B#C5#czS|>'
        'vcmUd96(19`G`>DV@?Q2nvI<d@JZ&SbENr|37v+nd8v0D3}Y%;HWXVp_BmdL5gQKEdO%?Ebe*w4Uq^$CqXEXgoP0'
        'zK@uT90eV<J@I?#v6ka>PbXOJiWRlYbX)Tw9x<e_0sNA4^=%De2Zm#x*NBTh{|e)8GV@<?rb_MQzl2AeWn&AvYGS'
        'YwtBXqV3^&!W9c>39|>_opHHejJbp8c@Crr(Lt%=+}T%eW02FEj%aPlDJlB4YUL@nM(EcSp0sMARtHiS={DT`lqX'
        'MTO{a+5)l^p1+YDfOGP4Efw!uG5RnuV4*#Bw5co{ffSNmJdaD30c!A=V5jl^^JB(CQx(!y+Rb3&`W&*L-'
        'eZwX;a463iLQ11)NfAL?1#<wX1TR#_L`~>CD_mgd=eq4MTutQ=z?}SIZ%EccT)4k<)OZjE|2=VF`G=8khdjQ_a0%'
        'a@8EEI{as^_DLFe@!0O?5|q`qEoHVTXG>!sT$$zTIp8WC+21F{d)3)!Y+q-'
        '?IIkjvt8HS|y(ETpzj2U~+PqD7m5Cd!zv?MY1RMC5R&lK`<{3EN+Ygd*;>J10r)+`>jUq_K0g9>r!7!TP>pRq4o8'
        '2i_IAgD+uS_rxmKvhO10aq*n-'
        'e6|(rOE*J<%$WZx>}md{M_xG#LC>8X{%3PX$AdOsf9ZJuowAblki)&#wBJ49>>2n$tkMR)bJDb(ia1!P<3NF>rBZ'
        'Iu<+HnOEOMiW;n17K!sAc)@F?>M%t3^Hr308TjOB^f$~_?)V(qTR9&<F>vX<r5k>q=}ilNtN9gx#e9`~~#9cZ}`-'
        '#BXB!qJGC*oM^+`%gMHW?+j8rWz>P%kFAvQH?EIG#RstTb-'
        '55UAizk^&<T+#Cwr9vn09^J<yZzXvBkNlY5?x8RR#oMBnuMHED6=JdM$V_1<qyl=#UVXb5zDE0Y~4@zCSE@K(w4z'
        ';$0zivHuZPK(DJr2UP@E0m5nUS^1B19cV?Y%cK}UlS?QSQByW<M70z8kBvF^zpK&<epyxCRn{&sOuTBrn;QD=7+1'
        'a3<e|>C=NS@p0`X=u?ciNkZOIQ?x$#H&*nlK%o7R+urK@+h=)qACC!9U(C1NZNn?okW{U#*XQ^r8jAceNOePsJ4?'
        'DBY3|Y?mGd+mT{W;Jn$|?5LLx1Fmn?yOL!-'
        'MJ?x|LcXfkKJK$fX11Fik#|9*$ZGh+$ER5X9VXvd>FA{Y&AMpw~nQr{1m3#6=4t$XoAY#6EvAcSXGpeWtb_kPd3V'
        'WQzC+;u)PxN;o=_5s@-ldb21Fxz#Sp*4KH}<$Pj=L7w@9z#}2*lS{yJ_xR1+9IAmCs}F+BYzuS-'
        '>=_>ddu2|6g^7{a&#j*LpwGm*^&Jc%QxsT7bRZUsdhbX%zqdWJo>EdcO)r-'
        'r%S56JfeO{3=ncgPmt4I>4tEeEm4m_aVka&$+D}t4(Dmg^MsSGAsqfAoi@Xn-cyJ`<!)%|C@W}^<ou%<1;z<=vXy'
        'go2+BZ!70Q{j!BYA^Rlp*1Vyi6Xh^RV?8WS5h|aU5Igdab6DUJH`?K#Ax*qQ*Q@jtv=naa+l&e0xhashdzE0CpG8'
        '^$eVYXZWN1BXX)tPmej=uJ&Uy(@DP?EZat+m{lmOnm>b(24<PcW}wUpBYykAMIm4~oXndPQwnZkC0A<o9NYiole^'
        'y305U~;c1O<H`$*Tl!$DC<cv&0P_YW5o3d<tm>RRv;*!NExtGcg4{b>{WyDHM<ROsW04$^F}dQ{A~Yhnyubf=vAe'
        'Gq=y`}Mv*2ZWsRy`>&13S_-dupU#Bgvs6gvH2zL2V{;qXN2K9O8PA5)y3_M<cZ4PtrE?u1K8>@Z+Xi>W%m{bri>v'
        '(@h!tC`rIWNZWAMCE4&t6oly#rsmdzkhk*tWDr{zPX&asrjy|Fm%{(^F?Ml$%CYf#r-'
        '<+p(j<>&>%AG>DC+IDm5Qt8oFaXH+HY%Wl{BQSwhU}8Fe!SG?Wp5)x2ph!{i=*f%o$)pEvX19P9{{0(iB3Is2z1('
        'cs*l``LDVNPS(jZFH76<6Imsvl0}p8r4uXqKOf`B4J&M{UJ-tlcL=BBfVFy81fzcvj!CB%xTk$?N!Q_8Wj|Cjqkb'
        '|Iyh!C+Dq_QxqF^IPwsX_DOeDZLeb2KB%*saqM4J>zmqLfxP$gw=;nI6VTH(+!OXxaWw4%r;zc1N%6@$ljHhke+h'
        'IUj|Uj+W39BUr->SG^zndYsuZW=IT7u=^4BouQUWNHT18yCshF7DzAmNE4C8qC_xc33r3FA7|kl96)+}M2=Y-'
        '+7|XrxyzedlX_qje-AltA<OB3jvj)3zrcg>3h5fbEH>l=2sv&o7o*q9esoVGZl8}u!yhxg8*J%0%z`RcAWVZgekX'
        '?-a++}P8*D2X=-9;Pv*UU?igmQFBlK~5dK}7;z8*s&bMlN41pRc-(b308)jI8R;)Wn$uPpO`SX-'
        'wah!a5?)Xnni5VNZdflRrp&9nAf_ih}-HxNQ$@AE-eg68Ye&f59#<aAM%d5}HiqbbxGrB$i&Zco2dbudI0U5D$b$'
        'M0c_bR2XLDqe={kmWwv<k$6~83xa2zdi=MPu2bg+ED^vWcBvY;tigzhuhgUh{)PyEXDQ(AIJ`KdV0&wUQ=}H%t<I'
        '_ZK!EXyc^4U+Uq{_q!WgS`|o<{hC&J|hcA0-'
        '_sI~XPtT_wNd5tUetXv6&>Pgt{RVZo^%qm4^N08aBTFDmsI7mG(a_W`cl)Nk*1`V$=J=rhYv5?uBeJI(scfynr{i'
        '|Vh>1LPjh(^1j;!ipJke|2%_qk?SfkJaIWhm2xlQ9%?-'
        '*GlOEQy~eT*AjRlrU&sX#|+k%X7EIQ3iKbYUq?`Iq@;Ps!NM`Sdipp*}zAj?dO^GWcKSy&Ff~@kGqZ(RXMux?+xM'
        '3}g7)GiyuFcQg-pa=@wM#*i-E$Np**BfZ@Bk8#Xr-k0}YpY`$<RZCCf_McW#j_etwoj-MU`fLXKo-'
        'k<6_MZKS1B!dr%FUB{&T|4Cah;o_?iK)T5}jfz)HwV-'
        ')lsnGPF*TxZ+Z{T^W5Q0)gOx6^i;yeU^PY!K7n}(kBA)sGj~1qw}ymB{=so5VUuUkhMYFe`(caQ4+F?PbxcOGbPy'
        'hehn1@%;31NO&^R(K0Hm@68|A06veE0*lc!O*c$>gKFboL(#b_V8yk0@|?QPkn$<v~4qvaN(_p~{uq=6~8sOCG+{'
        '&hnrxo02jrDM+FgRfC;h_edbSphk9dQS>Cg>1Nty5bqPp{DBUP|PsD%P)bZhc!L$2E<&Zzh7^tRMB14YbwI`X4uQ'
        '7tf-r--@0$)I@PzL1u5WgRGs1rORM-|C<w)&S&#0pJ~#`O3#)p63no*s7dP4|6x^QbOR-'
        'r}d8eG<)tOuT+`l|ZiAqZnEWtS71+=GgJqqUw++<WY273qW+#%6Jb>5dq>5MZe(P2NYkaOLKmoJJ^%&6T=FJV=`B'
        ';VyF*iJ-'
        '(tS#^Y&6zrmY2G74m(^SPtr^)p)Bn(%gbbHXd74%h>Q8KB%{fOdb2g<H({r6u8E;275{`I)3lO_<Tdrhj!R!G#w0'
        'Oe0xMtYT%oiL-6<F;EBZY?8-L?#+>Un)qX=o*T>Vnbv5V<xQY^KW^WOBYG$f`fBzN?uu<_-*{Vy!`od%96zV-'
        'O*KI18<;a}fEn@cbp}U9b*dBvPk`KFw$w_@CHH>YpPtW2{Jlvjq$GUDX_+r22Jy)d%s3y+TDFwc>bc49(~IX_^M<'
        'hL_g(QRt;!Mq5d5eIVMRz-'
        'lo>!W^B(#GE+G(eH4PH!VsG?iYFu=!Z%(%FW&Tgfpz`9pP7Tf(Tci#?AtM@b65~_(LU*_tn=z!)Q<agYD!@im-pk'
        '>o;PIZh$xeIV`yyX{?XE+hoQ%kk^gH5Y#mFT~E{lFfaoi=~`s)Bc;yFO*RkW19vPmQ)p20@FJaP)3SG79m_@jQ0m'
        'F9FP^Y)#TO?P?!>|5MkN39xp(5bR``k^JsPAzqc~4jrCIix8BWjVp7f`eG3oCD)2m?(jGsMV{Bi`C2@dheDW{PBD'
        'ZU0b9Eopgz#ZZjI}G7oEpR=JiIS%o%7;YWC^XfDdg_XjIFLovnbop12IM!%`LZR)bUKY_bN(c8(|5Evn0c49O#@Y'
        'fSvdT6UU?(=R9W*Jfy>=>jkLgvqja^N1xkiZ!ma1f$&A?m4-'
        '#nru;7;wmvmF)?2eYD`DN@1M|aF*9r>yE$~&ERGh+Ys4SCD}?6vKbb?ZqD_0}fE3i*;wu~qD9MZHAn@}}W3-'
        '4$V7*=>=ZR0E!X%Wuk-53r-;6ynA!*AyzZBirv#7;|QY^4G%ph7e6QWrvoR#n-'
        'Du;B2igxOhJzU|1GWKkM|Jt^+1I-'
        'guBy4hG6B{5PvWfMM4~yK2fkmBXc0anmK?@xUntaUNO#i?>IBWI^kTwLL8dRSWgSkJ!RuI((7W(CC7#(Up{-'
        '4MP?S+DSNee5ZG6i0_mC{l5|gDG`tchyGpB!T`1M#*A?uaAkai3BDV}!W;i1(CFuYur<CejJK|v5cHhZDY+iN>7;'
        'LO0~m3Xaem3T=d0CWci85X(F#*)s#1>~eF0CU9#!xLdEjldOgQY%MyCGfN4D;beZ2Yp;qGe&#|Lj-'
        'p^QtQm@uCX=f!OQZy9fM$y;W^I3GLI#cp5n>em2GEYRWcHdXc9_!Xx7+lS!ebAYMaTce*H-'
        'd~mEddRC`n($SAfh1>H5EK5IU096paQ<TM2v-Gt<Ll3gu}M2NVcqdM&UjuBa?Vm?Z1U=n;&D-'
        '#Iqd{BNYXWiI7N>j>5yr5e3qRf{m>8h-'
        '$nOxE^ae5oJU`E`&jof*&@o!lLKorN}9UjYvigD7jNK;P949W8Sr=asps?U^^`iBgtpADcKp%lHlWS(6c3f#E(Bf'
        'FyiR@~#`erE3^{cszgCb4syS|TFFlr&d`-'
        'hklh6E|XMxjl5}qPM!knn#(klcl2Y%PwCs#QlLhh)p#4P~Zfqd0q774!m<uoF~o+LPp7eCf`btFi<VlB{3BNWxb$'
        'uO>^<eB<=F3B#gz)^2nq>K1!0hrlKJM1|Jt!utC(^!18`<C(Bdwe;H{HQnZ8N$MiYC|g<%mZ?`Z~`e?Qhn<27RZt'
        '@Z^5?iGUb$Tfa>?Lb-n}BU*~Hp(7M=U@zCFZchP3~O%Wa=z-'
        '>BI9O<b{bEpzmS$_`Qwg<&zZXMaAX&_8e5_Q@{@z$<E6D#Pxm-'
        '`1%Uy^p?_kh=KXCJ)SSQrGaQ;itUO28#<2{{1tCR>;JWmVI?Gl8uVMLDPzkE!e{-'
        'Fu!e48M`cZU7!4cy>3{dK_OZ=wYr@ynaa!_dM%!)+rJvf^{w(etA=YZjI7UfAf7UQY2_eaFn+06%Se{9-'
        '(;;Reqh9TMpRp;xjP)w(y|;%6q`J*Ixbx#@^Gjp*xL(yp9&AVSt>?N+gO!tOki`AOKu9nxN(aY`5S5rn+1WEl@#7'
        '>*_=3%ExyfoFLT!ARc<6&0zW<xK+N-FUl=#S}aw}m+H+lz7;I4fa4#4qG$zPizm&-'
        '7l(4Y=BUv%n++MSG?)ULhK5UqEI7udPVD}Mw#~n7{!^q)6Nj+8`QVmT6Vcg`ibK@;I-YBuZes-'
        '=>aI>zM9Fu#zO4DEiK%v<)Hah_F~c3&qU~hcy5c+jL|36GDZ%&P`n`xX;BDGr75(A|OmP;eQtuT}f3WvvyrV(;mc'
        'RIwNa6&A-*IG$&i?#g_z7>XEw-BzT*ZKq2MZp%6jduwT9Gots8uFY-'
        'X?#h3vuyQY=tP`p?|@A=tEk~PBzWZM88(+p^RVIDY<DJ6G_G1`7w8b{g&eQTx_QchFdyMwvcAIrT!r<|E4is@}J('
        'DPnX`asL%ag`s9+i`L&;SnJ}&1@Uuq?oxLzln4X*%l0M<FN&jZozU+>MKAuTEaue*4*PHn}+RM=>(r{VklXo$&PT'
        'My%$i`hp+2DeT^6NazS;;}2K`rw_-'
        '$USqeJVCmJ(Jv&y0gis>?KAiZE@x5>Xg|;Mq&Y*h=f|YE=<6?`|!OHZ?a{X635ZyRtui`YR%ShR{FyRNYKSVUx$W'
        '>OiAb&l=ndj;Ydo_p&RM!ai-pEGJV%FJb3tHX>5<ZuhmJxi)a5yZmOSIXES8m&In$A9J@-'
        'USjq=mL{fe!A0%^$V8UfI3^QPgRNm{yJ7>!Qzgn!%0nA@RaA>_{4K(UeWO>ljC3TPdlI)e4uwzop-'
        ';h`9Lt8YnR1eL}#OZQTSSxNe<;{ZPBw$&rVkGaTi>q$8C2tD+nBEUvOePxT>&c3W;<gC{3Jn6Gvc#$GxEG89*X5w'
        'C^I$P=e8mAh>BHQQYqBynR!4q6#`Vh>NLG$P9b<ZvEG2`XK#z&Y_)1FGZxMaOI=G7JaXjbR9)##C;iE2+GMJPBvB'
        'cp1eYKn#6uNZi&>Mt?R9EhbMOA;C{qX)xx&ZCD!Ha6U8NP4##cDZiVYy#nTz-JXKKl?ILj2$r4yH-'
        '0;)t`<II?gxLuUrE@7{g;m#^M_oqhlI>(9S=|K*$S&jy)8=Esum9<xU3%f~)<c)DQ}Y}6RPlBKNd3V2^CO81z>;;'
        'N7W!J%gwrXH30lcTCpG1xfjAqSU~m{aFCg#6<4NG>+r0~F$mtbs9zChHYe5lKFrZ!rZMy`457jRfPRBy`5ip5+rb'
        '=(nMY5PCjp{)+Fh)F{WNRP5`QK`l&^_sfh~C)N({F{Xiu++F{e(O@wDN|eR56o_~4EftUAB>^n#!y_%DBdwz&EF`'
        'n76*+Mi84AmL<45$|bks)z!=nF!VZImEkJ`*w>^`i^W~O#FR93g*2Bv;ie-QEEsuH;oGFlNg7)8@$;wsS$ZI_u4U'
        'W#o%SAuy1GZar}Ob^}W<WKI^t393)%WlG2TJmpUN3w!gs=xMwmDWr9(=nRax>yd_7JWv;@+hMM4?SfWo=?O}=g-'
        '*j6Lgj=LbZE6?Q_=jerHnY&%NG7C~!XcKYwV<UH'
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
