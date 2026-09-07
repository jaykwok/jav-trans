"""Authored, non-private Japanese dialogue cases for translation experiments.

References state one acceptable meaning, not an exact-match answer key. Keep
the neighbours and measured-layout stand-ins identical across experimental
arms. These cases do not substitute for human review of real source material.
"""

CASES = [
    ("decline-help", "ambiguity", [
        ("荷物を持ちましょうか。", "要帮你拿行李吗？"),
        ("大丈夫です。自分で持てます。", "不用了 我自己拿得动"),
        ("そうですか。気をつけて。", "这样啊 小心点"),
        ("ありがとうございます。", "谢谢"),
    ]),
    ("reassurance", "ambiguity", [
        ("転んだの？けがはない？", "摔倒了吗？没受伤吧？"),
        ("大丈夫です。少し驚いただけ。", "没事 只是吓了一跳"),
        ("無理しないでね。", "别勉强自己"),
        ("うん、少し休む。", "嗯 我休息一下"),
    ]),
    ("decline-tea", "ambiguity", [
        ("お茶をもう一杯いかがですか。", "再来一杯茶吗？"),
        ("結構です。もう十分いただきました。", "不用了 已经喝够了"),
        ("では、お水だけ置いておきます。", "那我放杯水在这儿"),
        ("お気遣いありがとうございます。", "谢谢你这么周到"),
    ]),
    ("approve-plan", "ambiguity", [
        ("この予定で進めてもよろしいですか。", "可以按这个安排进行吗？"),
        ("それで結構です。", "这样就可以"),
        ("では、明日から始めます。", "那明天开始"),
        ("よろしくお願いします。", "拜托了"),
    ]),
    ("double-negation", "negation", [
        ("一緒に行きたくないの？", "不想一起去吗？"),
        ("行きたくないわけじゃない。", "并不是不想去"),
        ("ただ、今日は時間がないんだ。", "只是今天没时间"),
        ("じゃあ、また今度にしよう。", "那就改天吧"),
    ]),
    ("not-all", "negation", [
        ("みんなが賛成したの？", "大家都同意了吗？"),
        ("全員が賛成したわけではない。", "并不是所有人都同意"),
        ("まだ反対の人もいる。", "还有人反对"),
        ("もう一度話し合おう。", "再讨论一次吧"),
    ]),
    ("not-necessarily", "negation", [
        ("値段が高いなら、おいしいよね。", "贵的话就好吃吧"),
        ("高ければおいしいとは限らないよ。", "贵也不一定好吃"),
        ("安くてもいい店はある。", "便宜的好店也有"),
        ("じゃあ、おすすめを教えて。", "那给我推荐一家吧"),
    ]),
    ("only-one", "quantity", [
        ("切符は何枚残ってる？", "还剩几张票？"),
        ("一枚しか残っていない。", "只剩一张"),
        ("二人では入れないね。", "那两个人进不去呢"),
        ("次の回にしよう。", "等下一场吧"),
    ]),
    ("not-yet", "negation", [
        ("もう昼ごはんを食べた？", "已经吃午饭了吗？"),
        ("まだ食べていない。", "还没吃"),
        ("じゃあ、一緒に食べよう。", "那一起吃吧"),
        ("あと五分待って。", "再等五分钟"),
    ]),
    ("no-longer", "negation", [
        ("あの店にはよく行くの？", "经常去那家店吗？"),
        ("今はもう行っていない。", "现在已经不去了"),
        ("引っ越したから遠くなった。", "搬家以后离得远了"),
        ("近くで探すしかないね。", "只能在附近找了"),
    ]),
    ("conditional-rain", "condition", [
        ("明日は外で練習する？", "明天在外面练习吗？"),
        ("雨が降らなければ外でやる。", "不下雨就在外面练"),
        ("降ったら体育館を使おう。", "下雨就用体育馆吧"),
        ("朝、連絡するね。", "早上联系你"),
    ]),
    ("even-if", "condition", [
        ("雨だったら中止かな。", "下雨就取消吗？"),
        ("雨が降っても中止しない。", "即使下雨也不取消"),
        ("屋根があるから大丈夫。", "有顶棚 没关系"),
        ("それなら安心した。", "那我就放心了"),
    ]),
    ("before-after", "condition", [
        ("先に帰ってもいい？", "可以先回去吗？"),
        ("片付けが終わってから帰って。", "收拾完再回去"),
        ("終わる前に帰るのはだめだよ。", "没收拾完不能走"),
        ("分かった、手伝う。", "知道了 我来帮忙"),
    ]),
    ("unless", "condition", [
        ("連絡がなかったらどうする？", "没收到消息怎么办？"),
        ("連絡が来ない限り、ここで待つ。", "没收到消息就一直在这儿等"),
        ("来たらすぐ出発しよう。", "一收到就出发吧"),
        ("携帯を確認しておく。", "我留意手机"),
    ]),
    ("possibility", "modality", [
        ("明日は来られる？", "明天能来吗？"),
        ("仕事が早く終われば来られるかもしれない。", "工作结束得早的话也许能来"),
        ("まだ決まっていないんだね。", "还没确定是吧"),
        ("うん、分かったら連絡する。", "嗯 确定了联系你"),
    ]),
    ("hearsay", "modality", [
        ("駅前の店、閉まるんだって。", "听说车站前那家店要关门"),
        ("本当？まだ確認していないけど。", "真的吗？我还没确认"),
        ("隣の人から聞いただけ。", "我只是听邻居说的"),
        ("じゃあ、店に聞いてみよう。", "那问问店里吧"),
    ]),
    ("expected-not-certain", "modality", [
        ("資料はもう届いた？", "资料到了吗？"),
        ("昨日送ったから、今日届くはずだ。", "昨天寄的 应该今天到"),
        ("まだ届いていないみたい。", "好像还没到"),
        ("配送状況を調べるよ。", "我查一下配送情况"),
    ]),
    ("benefactive-received", "roles", [
        ("その本、どこで買ったの？", "那本书在哪买的？"),
        ("姉にもらったんだ。", "姐姐给我的"),
        ("誕生日のプレゼント？", "生日礼物吗？"),
        ("そう。まだ読んでいない。", "对 还没读"),
    ]),
    ("benefactive-gave", "roles", [
        ("妹の誕生日、何をあげた？", "妹妹生日你送了什么？"),
        ("前に欲しがっていた本をあげた。", "送了她之前想要的书"),
        ("喜んでくれた？", "她喜欢吗？"),
        ("その日のうちに読んでくれたよ。", "她当天就读了"),
    ]),
    ("causative-permission", "roles", [
        ("その写真、私にも見せて。", "那张照片也给我看看"),
        ("いいよ。でも持って帰らないでね。", "可以 但别拿走"),
        ("見るだけだから。", "我只是看看"),
        ("じゃあ、どうぞ。", "那给你"),
    ]),
    ("passive", "roles", [
        ("どうして遅れたの？", "为什么迟到了？"),
        ("駅で知らない人に道を聞かれた。", "在车站被陌生人问路"),
        ("案内してあげたんだ。", "你给他带路了啊"),
        ("うん、近くまで一緒に歩いた。", "嗯 一起走到了附近"),
    ]),
    ("borrow-lend", "roles", [
        ("傘を貸してくれる？", "能借我把伞吗？"),
        ("私も兄から借りているんだ。", "我这把也是向哥哥借的"),
        ("じゃあ、別の人に聞く。", "那我问别人吧"),
        ("ごめんね。", "抱歉"),
    ]),
    ("asked-to-wait", "roles", [
        ("どうしてここにいるの？", "怎么在这里？"),
        ("ここで待つように先生に言われた。", "老师让我在这里等"),
        ("私が先生を呼んでくるね。", "我去叫老师"),
        ("ありがとう、助かる。", "谢谢 帮大忙了"),
    ]),
    ("polite-request", "register", [
        ("すみません、窓を閉めていただけますか。", "不好意思 可以关一下窗吗？"),
        ("少し寒いですか。", "有点冷吗？"),
        ("はい、風が強くて。", "是的 风有点大"),
        ("すぐ閉めますね。", "这就关上"),
    ]),
    ("plain-request", "register", [
        ("窓、閉めてくれる？", "帮我关下窗好吗？"),
        ("寒いの？", "冷吗？"),
        ("うん、ちょっと。", "嗯 有一点"),
        ("分かった。", "好"),
    ]),
    ("urgent-repeat", "repetition", [
        ("もう電車が来たよ。", "车已经来了"),
        ("待って、待って。まだ荷物が。", "等等 行李还没拿好"),
        ("慌てなくていい。次もある。", "不用急 还有下一班"),
        ("そうだね、危なかった。", "也是 差点出事"),
    ]),
    ("hesitation", "register", [
        ("返事は決まった？", "想好怎么回复了吗？"),
        ("えっと、その、まだ考えていて。", "呃 那个 我还在考虑"),
        ("急がなくていいよ。", "不用急"),
        ("もう少し時間をください。", "请再给我一点时间"),
    ]),
    ("soft-disagreement", "register", [
        ("これで全部終わりだね。", "这样就全做完了吧"),
        ("そうとも言えないんじゃないかな。", "恐怕也不能这么说吧"),
        ("まだ確認が残っている。", "还需要检查"),
        ("忘れていた、ありがとう。", "我忘了 谢谢提醒"),
    ]),
    ("number-correction", "quantity", [
        ("集合は七時半だったね。", "七点半集合吧"),
        ("七時半じゃなくて、八時半だよ。", "不是七点半 是八点半"),
        ("一時間勘違いしていた。", "我记差了一个小时"),
        ("早すぎなくてよかったね。", "幸好没去太早"),
    ]),
    ("remaining-quantity", "quantity", [
        ("全部でいくつ必要？", "一共需要多少个？"),
        ("六つ。今四つあるから、あと二つ。", "六个 现在有四个 还差两个"),
        ("二つだけ買えばいいんだね。", "只要再买两个是吧"),
        ("うん、それで足りる。", "嗯 那就够了"),
    ]),
    ("comparison", "quantity", [
        ("電車とバス、どちらが早い？", "火车和公交哪个快？"),
        ("今日はバスより電車のほうが早い。", "今天火车比公交快"),
        ("道路が混んでいるから？", "因为路上堵车吗？"),
        ("そう。駅まで歩こう。", "对 走到车站去吧"),
    ]),
    ("not-more-than", "quantity", [
        ("歩いて何分くらい？", "走路大概要多久？"),
        ("十分もかからないと思う。", "我想用不了十分钟"),
        ("それなら歩けるね。", "那可以走过去"),
        ("近道を知っているよ。", "我知道近路"),
    ]),
    ("only-today", "scope", [
        ("いつでもこの値段ですか。", "一直都是这个价格吗？"),
        ("今日だけ半額です。", "只有今天半价"),
        ("明日は元の値段に戻ります。", "明天恢复原价"),
        ("じゃあ、今日買います。", "那我今天买"),
    ]),
    ("only-looking", "scope", [
        ("何かお探しですか。", "您想找什么？"),
        ("見ているだけです。", "我只是看看"),
        ("どうぞごゆっくり。", "请慢慢看"),
        ("ありがとうございます。", "谢谢"),
    ]),
    ("request-not-question", "speech-act", [
        ("そこにある箱、取ってもらえる？", "帮我拿一下那边的箱子好吗？"),
        ("この青い箱？", "这个蓝箱子吗？"),
        ("ううん、その隣の白いほう。", "不是 是旁边那个白的"),
        ("はい、どうぞ。", "来 给你"),
    ]),
    ("permission-vs-ability", "speech-act", [
        ("ここに座ってもいいですか。", "可以坐这里吗？"),
        ("どうぞ。誰も使っていません。", "请坐 没人用"),
        ("荷物は足元に置きますね。", "行李我放脚边"),
        ("そこなら邪魔になりません。", "放那儿不碍事"),
    ]),
    ("prohibition", "speech-act", [
        ("このボタンを押せばいい？", "按这个按钮就行吗？"),
        ("それは押さないで。", "别按那个"),
        ("右側の青いボタンを押して。", "按右边的蓝色按钮"),
        ("こっちだね。", "是这个吧"),
    ]),
    ("no-need", "speech-act", [
        ("明日も来なければいけませんか。", "明天也必须来吗？"),
        ("明日は来なくてもいいです。", "明天不用来"),
        ("次は来週の月曜日です。", "下次是下周一"),
        ("分かりました。", "明白了"),
    ]),
    ("must", "speech-act", [
        ("この書類、今日中でなくてもいい？", "这份文件不一定今天交吧？"),
        ("今日中に出さなければいけない。", "必须今天交"),
        ("締め切りが五時なんだ。", "截止时间是五点"),
        ("急いで仕上げる。", "我尽快完成"),
    ]),
    ("past-regret", "modality", [
        ("昨日、誘ってくれればよかったのに。", "昨天要是叫上我就好了"),
        ("忙しいと思っていた。", "我以为你忙"),
        ("昨日は休みだったんだ。", "我昨天休息"),
        ("次は必ず声をかけるよ。", "下次一定叫你"),
    ]),
    ("counterfactual", "condition", [
        ("もう少し早く出れば間に合ったのに。", "要是早点出门就赶得上了"),
        ("道がこんなに混むとは思わなかった。", "没想到路上这么堵"),
        ("次は電車で来よう。", "下次坐火车来吧"),
        ("そうしよう。", "就这么办"),
    ]),
    ("referent-object", "reference", [
        ("赤い袋と青い袋があるね。", "有红袋子和蓝袋子呢"),
        ("赤いほうを先に開けて。", "先打开红的"),
        ("こっちは後でいいの？", "这个可以待会儿再开吗？"),
        ("うん、青いほうは後で。", "嗯 蓝的待会儿再开"),
    ]),
    ("unknown-person", "reference", [
        ("誰から電話だった？", "谁打来的电话？"),
        ("名前は聞き取れなかった。", "名字没听清"),
        ("あとでまた電話すると言っていた。", "对方说待会儿再打"),
        ("分かった、待っていよう。", "知道了 等着吧"),
    ]),
    ("relative-clause", "continuation", [
        ("どの店にする？", "去哪家店？"),
        ("昨日話していた", "昨天说过的"),
        ("あの店に行こう。", "那家店吧"),
        ("いいね、予約しておく。", "好 我先预约"),
    ]),
    ("split-negation", "continuation", [
        ("一緒に来るのは嫌なの？", "不愿意一起来吗？"),
        ("行きたくない", "并不是"),
        ("わけじゃない。", "不想去"),
        ("今日は少し疲れているだけ。", "只是今天有点累"),
    ]),
    ("split-condition", "continuation", [
        ("いつ手伝ってもらえる？", "什么时候能来帮忙？"),
        ("明日の仕事が早く終わったら", "如果明天工作结束得早"),
        ("そちらに寄るよ。", "我就过去一趟"),
        ("無理はしないでね。", "别勉强"),
    ]),
    ("split-not-all", "continuation", [
        ("全部間違っていたの？", "全都错了吗？"),
        ("全部が間違っていた", "并不是全都"),
        ("わけではありません。", "错了"),
        ("最初の部分は合っています。", "开头的部分是对的"),
    ]),
    ("ordinary-greeting", "control", [
        ("おはようございます。", "早上好"),
        ("今日はいい天気ですね。", "今天天气真好"),
        ("駅まで一緒に歩きませんか。", "一起走到车站好吗？"),
        ("ええ、行きましょう。", "好 走吧"),
    ]),
    ("ordinary-meal", "control", [
        ("夕飯は何にしよう。", "晚饭吃什么呢"),
        ("冷蔵庫に野菜があるよ。", "冰箱里有蔬菜"),
        ("じゃあ、スープを作ろう。", "那做汤吧"),
        ("パンも焼くね。", "我再烤点面包"),
    ]),
    ("ordinary-directions", "control", [
        ("図書館はどこですか。", "图书馆在哪？"),
        ("この道をまっすぐ進んでください。", "沿着这条路直走"),
        ("角を右に曲がると見えます。", "在拐角右转就能看到"),
        ("ありがとうございます。助かりました。", "谢谢 帮大忙了"),
    ]),
]


def build_cases():
    rows = []
    for scene_index, (scene_id, category, lines) in enumerate(CASES):
        for offset, (ja, reference) in enumerate(lines):
            row = {
                "start": float(scene_index * 30 + offset * 3),
                "end": float(scene_index * 30 + offset * 3 + 2.8),
                "text": ja, "case_id": scene_id, "category": category,
                "reference": reference,
            }
            if category == "continuation":
                if offset == 1:
                    row["continues_into_next"] = True
                if offset == 2:
                    row["continues_from_previous"] = True
            rows.append(row)
    return rows


def plan_source_cases(rows):
    """Run production source planning on explicitly synthetic timings.

    Continuation fixtures describe a single original sentence. Their reference
    translations describe the parent source, not exact-match targets per cue.
    This fixture measures no audio alignment.
    """
    from llm.context import SourceContext
    from subtitles.options import SubtitleOptions
    from subtitles.writer import prepare_srt_blocks

    planned = []
    for unit in SourceContext.build(rows).units:
        members = [rows[index] for index in unit]
        text = "".join(row["text"] for row in members)
        words = []
        for row in members:
            step = (row["end"] - row["start"]) / max(1, len(row["text"]))
            words.extend({
                "word": char, "start": row["start"] + index * step,
                "end": row["start"] + (index + 1) * step,
                "timestamp_kind": "ctc_forced_alignment",
            } for index, char in enumerate(row["text"]))
        cues = prepare_srt_blocks([{
            "text": text, "start": members[0]["start"], "end": members[-1]["end"], "words": words,
        }], options=SubtitleOptions(drop_vocalisation_only_cues=False, timing_polish_enabled=False))
        for cue in cues:
            planned.append({
                **{key: cue[key] for key in ("text", "start", "end", "continues_from_previous", "continues_into_next")},
                "case_id": members[0]["case_id"], "category": members[0]["category"],
                "reference": "".join(row["reference"] for row in members), "reference_source": text,
            })
    return planned


def build_review_fixture():
    """Known meaning errors and correct controls; no first-pass API bill."""
    items = [
        ("明日は行かない。", "明天去", "明天不去", "negation"),
        ("三人だけ来ました。", "来了四个人", "只来了三个人", "quantity"),
        ("雨が降っても中止しない。", "下雨就取消", "即使下雨也不取消", "condition"),
        ("私は妹に本をあげた。", "妹妹给了我一本书", "我给了妹妹一本书", "roles"),
        ("昨日送ったと聞いたが、まだ届いていません。", "明天会寄出，已经到了", "听说昨天寄出了，但还没到", "meaning"),
        ("この店は今日だけ半額です。", "这家店每天半价", "这家店只有今天半价", "quantity"),
        ("明日は行かない。", "明天不去", "明天不去", "control"),
        ("三人だけ来ました。", "只来了三个人", "只来了三个人", "control"),
    ]
    return [
        {"text": ja, "start": float(index * 30), "end": float(index * 30 + 3),
         "case_id": f"review-{index}", "category": category,
         "reference": reference, "initial_translation": current}
        for index, (ja, current, reference, category) in enumerate(items)
    ]
