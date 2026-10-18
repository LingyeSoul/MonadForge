import type { CurationField, CurationStage } from '../types/curation'

// Presentation only: keys and option values remain the upstream request contract.
// Boolean copy describes the destination value, not a potentially inverted CLI flag.
type Copy = readonly [cnLabel: string, cnHelp: string, enLabel: string, enHelp: string]
export const curationFieldCopyCatalog: Record<string, Copy> = {
  src: ['原始图片目录', '原图及原标注所在目录；只有导出会回写修订原标注。', 'Source images', 'Original images and captions. Only export publishes revised master captions here.'],
  dst: ['缩放工作目录', '各阶段读取缩放图，并在此保存修订标注。', 'Resized workspace', 'Processing reads resized images and saves revised captions here.'],
  path_pattern: ['图片路径筛选', '按相对路径匹配通配符；多个条件用 | 分隔。', 'Image path filter', 'Match relative paths with wildcards; separate alternatives with |.'],
  target_res: ['目标分辨率档位', '每行一个档位：512、768、896、1024、1280 或 1536；需与训练设置一致。', 'Resolution tiers', 'One tier per line: 512, 768, 896, 1024, 1280 or 1536. Match the training settings.'],
  recursive: ['包含子文件夹', '递归处理子目录，并保持相对目录结构。', 'Include subfolders', 'Process subfolders and preserve their relative layout.'],
  overwrite: ['重新缩放已有图片', '已达到目标尺寸的图片也会重新处理。', 'Resize existing images again', 'Process images even when they already match their target size.'],
  workers: ['并行工作数', '用于图片处理或读写的并行工作数。', 'Parallel workers', 'Number of workers for image processing or file I/O.'],
  resize_crop_anchor: ['裁剪对齐位置', '缩放后的少量边缘裁剪以此位置为基准。', 'Crop alignment', 'Anchor used for the small residual crop after resizing.'],
  resize_crop_margins: ['预裁剪边距（百分比）', '按上、右、下、左顺序每行填写一个百分比；留空表示不裁剪。', 'Pre-crop margins (%)', 'One percentage per line: top, right, bottom, left. Leave empty for no crop.'],
  freefit_max_ratio: ['最大宽高比', '超过此比例时裁剪到上限；4 表示允许 1:4 至 4:1。', 'Maximum aspect ratio', 'Crop ratios beyond this limit. 4 allows 1:4 through 4:1.'],
  skip: ['跳过的图片', '每行一个相对原图目录的完整路径，使用 /，不支持通配符。', 'Images to skip', 'One exact relative source path per line, using /. Wildcards are not supported.'],
  excluded_dir: ['已排除图片目录', '读取排除记录，防止后续处理重新纳入这些图片。', 'Excluded images', 'Read exclusion records so later processing does not restore excluded images.'],
  report_dir: ['报告目录', '保存此阶段的处理报告。', 'Report directory', 'Save this stage’s processing reports here.'],
  apply: ['直接应用修改', '关闭时只生成预览报告，开启后写入处理结果。', 'Apply changes during the run', 'When disabled, produce a preview report only.'],
  tagger_dir: ['自动打标模型目录', 'Anima Tagger 模型所在目录。', 'Tagger model directory', 'Directory containing the Anima Tagger model.'],
  device: ['计算设备', '留空自动选择，也可指定 cuda、mps 或 cpu。', 'Compute device', 'Leave empty for automatic selection, or use cuda, mps or cpu.'],
  mode: ['打标方式', '可仅补齐缺失标注、追加新标签，或替换现有标注。', 'Tagging mode', 'Fill missing captions, append new tags, or replace existing captions.'],
  min_confidence: ['标签最低置信度', '范围 0–1；0 使用模型自身阈值，数值越高筛选越严格。', 'Minimum tag confidence', '0–1. Zero uses the model’s own thresholds; higher values are stricter.'],
  batch_size: ['每批图片数', '一次送入模型的图片数量；显存不足时可降低。', 'Images per batch', 'Number of images processed together. Reduce when memory is limited.'],
  from_report: ['已有报告', '按已审阅报告应用修改；跳过报告生成后已变化的标注。', 'Existing report', 'Apply reviewed proposals, skipping captions changed since the report.'],
  crops: ['保存主体裁剪图', '将检测到的主体另存为裁剪图，便于人工审阅。', 'Save subject crops', 'Save detected subjects as separate crops for review.'],
  flatten: ['合并位置子句', '将所有位置子句还原为平铺标签，包括手写子句；无需运行模型。', 'Flatten position clauses', 'Merge all clauses, including handwritten ones, into flat tags without running models.'],
  prompt: ['主体检测提示词', '用于 SAM3 主体检测的文本，例如 girl。', 'Subject detection prompt', 'Text used by SAM3 to detect a subject, such as girl.'],
  prompt_embed: ['主体软提示文件', '用已训练的软提示替代主体文本；填 none 则仅使用文本。', 'Subject soft prompt', 'Use a learned prompt instead of subject text. Enter none for text only.'],
  checkpoint: ['SAM3 模型文件', 'SAM3 权重文件的位置。', 'SAM3 model file', 'Location of the SAM3 model weights.'],
  score_threshold: ['主体最低置信度', '提高后会减少检测候选，也可能漏掉较难识别的主体。', 'Subject confidence threshold', 'Higher values reduce detections and may miss harder subjects.'],
  retry_score_threshold: ['补检置信度', '检测数量不足时，使用此阈值再次检测。', 'Retry confidence threshold', 'Threshold used when retrying an image with too few detections.'],
  part_prompts: ['局部补检提示词', '主体数量不足时尝试；用逗号分隔，留空关闭。', 'Body-part retry prompts', 'Comma-separated prompts used when too few subjects are found. Empty disables retries.'],
  part_score_threshold: ['局部检测置信度', '身体局部检测框需要达到的最低置信度。', 'Body-part confidence threshold', 'Minimum confidence required for a body-part detection.'],
  part_containment_threshold: ['局部框包含阈值', '局部框与已保留主体重叠超过此比例时，将其视为同一主体。', 'Body-part containment threshold', 'Suppress a part box nested this far inside a retained subject.'],
  iou_threshold: ['重复检测重叠阈值', '两个检测框的交并比超过此值时，按同一主体去重。', 'Duplicate overlap threshold', 'Treat boxes above this intersection-over-union threshold as one subject.'],
  containment_threshold: ['检测框包含阈值', '抑制嵌套检测框；大于 1 时关闭，以免误删重叠人物。', 'Box containment threshold', 'Suppress nested boxes. Values above 1 disable this to retain overlapping people.'],
  mask_containment_threshold: ['遮罩包含阈值', '根据主体遮罩的嵌套程度去重；大于 1 时关闭。', 'Mask containment threshold', 'Remove detections with nested masks. Values above 1 disable this filter.'],
  dedupe_fill_ratio: ['去重遮罩质量比', '重复候选按遮罩质量择优；0 表示仅比较检测置信度。', 'Duplicate mask quality ratio', 'Use mask quality to choose duplicate detections. Zero uses confidence only.'],
  min_area_frac: ['最小主体面积占比', '过滤小于整图此比例的主体，例如画中手机屏幕内的人物。', 'Minimum subject area', 'Ignore subjects smaller than this fraction of the image.'],
  pad: ['裁剪外扩比例', '在主体检测框周围保留的额外边距。', 'Crop padding', 'Extra margin around each detected subject.'],
  row_tol: ['行列对齐重叠比例', '主体投影至少重叠此比例时，归入同一行或列。', 'Row and column overlap', 'Minimum overlap needed to place subjects in the same row or column.'],
  max_instances: ['最多主体数', '每张图片最多绑定的主体数量。', 'Maximum subjects', 'Maximum number of subjects bound in one image.'],
  multiview_threshold: ['多视图判定阈值', '整图的多视图概率达到此值后，纳入审阅依据。', 'Multiview confidence threshold', 'Use the whole-image multiview prediction when it reaches this threshold.'],
  identity_confidence: ['身份特征置信度', '角色身份特征需达到此置信度，才参与多视图判定。', 'Identity confidence', 'Minimum confidence for identity attributes used in multiview decisions.'],
  suggest_counts: ['建议修正人物数量', '为额外人物提出数量标签建议；请人工核对是否包含男性。', 'Suggest person-count corrections', 'Propose count tags for extra characters; manually check whether detections include males.'],
  apply_verdicts: ['允许应用的判定', '用逗号分隔：multiple views（多视图）、extra-character（额外人物）。', 'Allowed verdicts', 'Comma-separated verdicts: multiple views and extra-character.'],
  apply_confidence: ['允许应用的可信等级', 'strong 为强证据，weak 为弱证据；多个值用逗号分隔，弱证据须先审阅裁剪图。', 'Allowed confidence levels', 'Comma-separated strong or weak. Review crops before accepting weak evidence.'],
  sheets: ['生成审阅联系表', '保存原图、主体裁剪和修改建议的对照图。', 'Create review sheets', 'Save comparison sheets with the original image, subject crops and proposed edit.'],
  multiview_audit: ['前置多视图审阅', '可关闭、仅生成审阅报告，或先应用符合条件的判定再继续。', 'Multiview pre-review', 'Disable, report only, or apply allowed verdicts before position processing.'],
  blank_crops: ['遮去裁剪图背景', '打标前遮去主体以外的区域，减少背景对人物特征的干扰。', 'Mask crop backgrounds', 'Hide areas outside each subject before tagging its crop.'],
  min_instances: ['最少主体数', '达到此数量的图片才生成位置子句。', 'Minimum subjects', 'Minimum subject count required to generate position clauses.'],
  strict_count: ['严格核对人物数量', '检测数量须与标注中的人物数量一致，才提出位置子句。', 'Require matching subject counts', 'Only propose clauses when detections agree with the caption’s subject count.'],
  max_clause_tags: ['每条子句最多标签数', '限制单个位置子句的标签数量。', 'Maximum tags per clause', 'Limit the number of tags in each position clause.'],
  max_novel_tags: ['最多新增标签数', '限制子句引入原标注中不存在的标签；0 表示不新增。', 'Maximum new tags', 'Limit tags absent from the original caption. Zero prevents new tags.'],
  name_confidence: ['角色名最低置信度', '角色名需要达到此置信度才会写入建议。', 'Character name confidence', 'Minimum confidence required to include a character name.'],
  allow_unlisted_names: ['允许新增角色名', '允许写入原标注中不存在的角色名，可能引入误识别。', 'Allow unlisted character names', 'Allow character names absent from the original caption; these may be crop misidentifications.'],
  discriminative_only: ['子句仅保留差异特征', '所有主体共有的特征留在平铺标签中，避免重复。', 'Use distinctive clause tags only', 'Keep shared attributes in the flat caption instead of repeating them in each clause.'],
  bag_gated_identity: ['身份特征以原标注为准', '子句中的发色、眼色须已出现在原标注中。', 'Require captioned identity traits', 'Only use hair and eye colors already present in the original caption.'],
  multi_view_gate: ['多视图共享身份特征', '同一人物的不同视图共享姓名与特征，不分配到各个视图子句。', 'Share identity across repeated views', 'Keep a repeated character’s name and identity traits in the shared caption.'],
  bind_view_anatomy: ['绑定视图可见身体部位', '按各个视图实际可见的身体部位生成子句。', 'Bind visible body parts to views', 'Place visible body-part tags in the clause for each view.'],
  bind_framing: ['绑定视图取景方式', '在对应子句中保留特写、全身等取景标签。', 'Bind framing to views', 'Include close-up, full-body and other framing tags in their clauses.'],
  rewrite: ['将专属标签移入子句', '从平铺标注移除已明确归属某主体的标签，避免重复。', 'Move attributed tags into clauses', 'Remove tags from the flat caption when their subject is unambiguous.'],
  bag_relax: ['已有标签阈值倍率', '原标注已包含的标签可降低阈值后归属主体；1 表示不放宽。', 'Existing-tag threshold multiplier', 'Relax thresholds for tags already in the caption. 1 disables relaxation.'],
  bag_word_relax: ['多词标签额外倍率', '标签每多一个词，额外乘以此倍率；1 表示不放宽。', 'Multi-word threshold multiplier', 'Apply this multiplier for each extra word in an existing tag. 1 disables it.'],
  bag_relax_min_score: ['放宽后最低置信度', '即使放宽阈值，也须达到此概率；0 表示不设下限。', 'Minimum relaxed confidence', 'Absolute confidence floor for relaxed tags. Zero disables the floor.'],
  attribution_margin: ['主体归属优势阈值', '获选主体须比其他主体更明确，标签才从平铺标注移入子句。', 'Attribution margin', 'Require a clear winning subject before moving a tag out of the flat caption.'],
  qwen3: ['Qwen3 分词器目录', '用于标注词元统计或标签随机替换，仅加载分词器。', 'Qwen3 tokenizer directory', 'Tokenizer used for token counting or randomized tags; no language model is loaded.'],
  max_tokens: ['报告词元预算', '报告按此上限检查标注长度。', 'Report token budget', 'Token limit used to assess caption length in the report.'],
  tag_csv: ['标签分类表', 'Danbooru 标签分类 CSV；留空使用默认查找位置。', 'Tag taxonomy CSV', 'Danbooru tag taxonomy file. Leave empty to use the default lookup.'],
  caption_insert_no_artist: ['补齐无画师标记', '没有画师标记时，在画师位置插入 @no-artist。', 'Insert missing artist marker', 'Insert @no-artist when a caption has no artist marker.'],
  caption_trigger_word: ['触发词', '将指定标签移动到标注中的触发词位置。', 'Trigger word', 'Move this tag to the trigger-word position in the caption.'],
  caption_trigger_at_front: ['触发词置于最前', '将触发词放在标注开头，而非画师位置。', 'Place trigger word first', 'Place the trigger word at the start rather than the artist slot.'],
  caption_drop_groups: ['要移除的标签组', '用逗号分隔组名或分类路径前缀，例如 artist、character、copyright。', 'Tag groups to remove', 'Comma-separated group names or taxonomy prefixes, such as artist, character or copyright.'],
  no_correct: ['保留原始标签顺序', '不重新排序标签；仍会移除指定标签组并生成所需变体。', 'Preserve original tag order', 'Skip tag reordering while still removing selected groups and creating variants.'],
  caption_shuffle_variants: ['标注变体数量', '0 表示不生成；第一个版本保留校正顺序，其余版本打乱标签。', 'Caption variant count', 'Zero disables variants. The first keeps corrected order; later variants shuffle tags.'],
  caption_tag_dropout_rate: ['变体标签丢弃概率', '仅对打乱后的变体生效；没有变体时忽略。', 'Variant tag dropout', 'Dropout probability for shuffled variants. Ignored when no variants are generated.'],
  caption_tag_randomize_rate: ['变体标签随机替换概率', '至少需 2 个变体及 Qwen3、T5 分词器，额外生成随机替换版本。', 'Variant tag randomization', 'Requires at least two variants and Qwen3/T5 tokenizers; creates randomized versions.'],
  t5_tokenizer_path: ['T5 分词器目录', '用于标签随机替换的 T5 分词器文件目录。', 'T5 tokenizer directory', 'T5 tokenizer files used for tag randomization.'],
  groups: ['要移除的标签组', '从修订标注移除选中类别；不会删除分类表中未知的标签。', 'Tag groups to remove', 'Remove selected categories from revised captions, retaining unknown tags.'],
  keep_tags: ['始终保留的标签', '用逗号分隔；即使属于待移除组，也会保留。', 'Tags to always retain', 'Comma-separated tags to keep even when their group is removed.'],
  category_paths: ['额外分类路径前缀', '用逗号分隔标签分类表中的路径前缀，用于移除特定子类。', 'Additional taxonomy prefixes', 'Comma-separated taxonomy path prefixes for removing specific subcategories.'],
  ocr_dir: ['OCR 结果目录', '按图片相对目录保存或读取文字识别结果。', 'OCR result directory', 'Read or save OCR results matching the relative image paths.'],
  min_chars: ['最少有效字符数', '不足此数量的识别结果会被过滤，不计空白字符。', 'Minimum text length', 'Discard shorter reads; whitespace does not count.'],
  skip_en: ['过滤纯 ASCII 文本', '过滤仅含英文、数字等 ASCII 字符的行，例如页码和网址。', 'Filter ASCII-only lines', 'Remove ASCII-only lines such as page numbers and URLs.'],
  strip_symbols: ['移除装饰符号', '移除爱心、星星和表情等装饰字符，保留常规标点。', 'Remove decorative symbols', 'Remove hearts, stars and emoji while retaining regular punctuation.'],
  det_conf: ['文本框检测阈值', '范围 0–1；达到此置信度的检测框才进入识别。', 'Text-box detection threshold', '0–1. Only boxes at or above this confidence are read.'],
  min_det: ['文本框保留阈值', '识别后，仅保留检测框置信度达到此值的文字。', 'Text-box retention threshold', 'Keep reads only when their detector box confidence reaches this threshold.'],
  min_box_px: ['文本框最小长边（像素）', '过滤长边小于此值的框，减少网点和纹理误识别。', 'Minimum box length (px)', 'Discard boxes with a shorter longest edge to reduce texture misreads.'],
  max_boxes: ['每图最多文本框数', '优先识别较大的文本框，限制单图处理量。', 'Maximum boxes per image', 'Read larger boxes first and limit the total workload per image.'],
  vl_batch_size: ['每批文字裁剪数', '一次送入文字识别模型的裁剪图片数量。', 'OCR crops per batch', 'Number of crops processed by the text reader at once.'],
  min_score: ['文字识别置信度', '有置信度的识别结果须达到此值才保留。', 'Text recognition confidence', 'Retain scored reads only when their mean token confidence reaches this value.'],
  mask_dir: ['补充文本遮罩目录', '可选；识别文本框未覆盖的遮罩区域。', 'Additional text masks', 'Optional mask tree used to read text missed by the detector.'],
  comp_min_side: ['遮罩区域最小短边（像素）', '仅用于补充文本遮罩；忽略更小的区域。', 'Minimum mask component side (px)', 'Ignore smaller components in the additional text masks.'],
  comp_max: ['最多补充遮罩区域数', '每张图片最多识别的未覆盖遮罩区域，优先较大区域。', 'Maximum additional mask components', 'Maximum uncovered mask components read per image, largest first.'],
  source_dir: ['待分组图片目录', '默认使用与其他处理阶段一致的缩放图片。', 'Images to group', 'Use the resized image tree shared by the other processing stages.'],
  out: ['输出位置', '分组清单或导出数据的保存位置。', 'Output location', 'Location for the group manifest or exported dataset.'],
  cell_match_min: ['局部网格匹配阈值', '网格单元的余弦相似度达到此值才算匹配。', 'Grid-cell similarity threshold', 'Minimum cosine similarity for a matching grid cell.'],
  match_frac_min: ['最小网格匹配比例', '图片至少有此比例的网格匹配才会归组；越高越严格。', 'Minimum matching grid fraction', 'Fraction of matching cells needed to connect images. Higher values are stricter.'],
  sim_min: ['整图相似度初筛', '先按整图相似度筛选，再用局部网格确认分组。', 'Whole-image similarity prefilter', 'Prefilter image pairs before validating their local grid matches.'],
  grid: ['网格边长', '每张图片划分为此数值 × 此数值的网格。', 'Grid edge length', 'Divide each image into a square grid with this many cells per side.'],
  ratio: ['匹配独特性比例', '数值越低，局部匹配筛选越严格。', 'Match distinctiveness ratio', 'Lower values enforce more distinctive local matches.'],
  min_size: ['每组最少图片数', '不足此数量的组不保留；1 表示保留单张图片组。', 'Minimum group size', 'Discard smaller groups. Use 1 to keep single-image groups.'],
  embedder: ['自定义特征提取器', '留空使用 PE-Spatial；自定义时填写模块与工厂函数。', 'Custom feature extractor', 'Leave empty for PE-Spatial, or specify a module and factory function.'],
  num_workers: ['图片解码工作数', '并行读取和解码图片的工作进程数量。', 'Image decoding workers', 'Number of workers for reading and decoding images.'],
  image_dir: ['输入图片目录', '生成遮罩所使用的图片目录。', 'Input images', 'Directory containing images used for mask generation.'],
  force: ['重新生成已有遮罩', '覆盖已有遮罩并重新计算。', 'Regenerate existing masks', 'Recompute and overwrite masks that already exist.'],
  masks: ['遮罩区域', '保留区域参与训练，忽略区域排除；可使用文本提示或软提示文件。', 'Mask regions', 'Keep regions participate in training; ignore regions are excluded. Use text or learned prompts.'],
  threshold: ['遮罩检测置信度', 'SAM3 检测结果需达到此值才生成遮罩。', 'Mask detection confidence', 'Minimum SAM3 confidence required to generate a mask.'],
  dilate: ['遮罩外扩（像素）', '向外扩展遮罩边缘；0 表示不扩展。', 'Mask dilation (px)', 'Expand mask edges by this many pixels. Zero disables dilation.'],
  mask_dirs: ['待合并遮罩目录', '每行一个目录，可同时加入自动生成和手工制作的遮罩。', 'Mask directories to merge', 'One directory per line; generated and handmade masks can be combined.'],
  output_dir: ['合并遮罩输出目录', '保存合并后的遮罩。', 'Merged mask output directory', 'Save merged masks here.'],
  master: ['修订原标注目录', '导出时将这里的修订原标注回写原始图片目录。', 'Revised master captions', 'Export publishes these revised captions back to the source image directory.'],
  index: ['标注索引文件', '要随训练数据一起导出的标注版本索引。', 'Caption index file', 'Caption version index to publish with the training dataset.'],
  combine_ocr: ['合并 OCR 文字', '把识别到的对白和拟声词附加到导出标注，工作区标注保持原样。', 'Include OCR text', 'Append recognized dialogue and sound effects to exported captions only.'],
  ocr_min_det: ['导出 OCR 检测阈值', '仅导出检测置信度达到此值的文本；0 表示不过滤。', 'Export OCR detection threshold', 'Include text at or above this detector confidence. Zero includes all lines.'],
  ocr_min_glyph: ['导出 OCR 最小字形（像素）', '过滤缩放图中过小的字形；0 表示不过滤。', 'Minimum exported OCR glyph size (px)', 'Filter glyphs too small in the resized image. Zero disables this filter.'],
  resize_cap: ['限制导出图片尺寸', '超过词元上限的图片会保持比例缩小，其余按原图输出。', 'Limit exported image size', 'Downscale images above the token limit while preserving their aspect ratio.'],
  resize_cap_tokens: ['导出图片词元上限', '4200 约为 1024² 像素；每个词元对应 256 像素。', 'Export image token limit', '4200 is about 1024² pixels. Each token represents 256 pixels.'],
  webp: ['以 WebP 格式导出', '以质量 95 转为 WebP；无需变更的现有 WebP 直接复制。', 'Export as WebP', 'Use WebP quality 95; copy existing WebP files unchanged when no resizing is needed.'],
  sidecars_only: ['仅导出标注与遮罩', '不复制图片或排除目录；不能与尺寸限制、WebP 转换同时使用。', 'Export captions and masks only', 'Do not copy images or exclusions. Cannot be combined with resizing or WebP conversion.'],
  caption_format: ['标注导出格式', '始终保留 Anima 文本；可额外生成 MOSS 布局标注，需标签分类表。', 'Caption export format', 'Always retain Anima text; optionally add MOSS layout captions using the tag taxonomy.'],
}

const isChinese = (language: string) => language === 'cn' || language.startsWith('zh')
const humanize = (value: string) => value.replace(/^--/, '').replace(/[_-]+/g, ' ')
export function curationFieldCopy(language: string, field: CurationField) {
  const copy = curationFieldCopyCatalog[field.dest]
  if (!copy) return { label: humanize(field.label || field.dest), help: '' }
  return isChinese(language) ? { label: copy[0], help: copy[1] } : { label: copy[2], help: copy[3] }
}

const groupLabels: Record<string, readonly [string, string]> = {
  detection: ['主体检测', 'Subject detection'], verdict: ['判定与应用条件', 'Verdict and application rules'],
  'multiview audit': ['多视图审阅', 'Multiview review'], 'clause composition': ['位置子句', 'Position clauses'],
  Detector: ['文本框检测', 'Text detection'], 'VL reader': ['文字识别', 'Text recognition'],
  'Combine OCR': ['OCR 文字合并', 'Include OCR text'], 'Resize cap': ['导出尺寸限制', 'Export size limit'],
}
export function curationFieldGroup(language: string, group: string): string {
  return groupLabels[group]?.[isChinese(language) ? 0 : 1] ?? humanize(group)
}

const choiceLabels: Record<string, Record<string, readonly [string, string]>> = {
  mode: { missing: ['仅补齐缺失标注', 'Fill missing captions'], merge: ['追加新标签', 'Append new tags'], overwrite: ['替换现有标注', 'Replace captions'] },
  multiview_audit: { off: ['关闭', 'Off'], report: ['仅生成报告', 'Report only'], apply: ['应用符合条件的判定', 'Apply allowed verdicts'] },
  caption_format: { anima: ['Anima 文本', 'Anima text'], layout_json: ['Anima 文本 + MOSS 布局', 'Anima text + MOSS layout'] },
  resize_crop_anchor: {
    top_left: ['左上', 'Top left'], top: ['上方', 'Top'], top_right: ['右上', 'Top right'],
    left: ['左侧', 'Left'], center: ['居中', 'Center'], right: ['右侧', 'Right'],
    bottom_left: ['左下', 'Bottom left'], bottom: ['下方', 'Bottom'], bottom_right: ['右下', 'Bottom right'],
  },
  groups: Object.fromEntries([
    ['artist', '画师'], ['character', '角色'], ['copyright', '作品'], ['meta', '元信息'], ['count', '人数'],
    ['clothing', '服装'], ['accessory', '配饰'], ['action', '动作'], ['pose', '姿势'], ['expression', '表情'],
    ['face', '面部'], ['hair', '头发'], ['body', '身体'], ['background', '背景'], ['effect', '效果'],
    ['object', '物品'], ['animal', '动物'], ['color', '颜色'], ['person', '人物'], ['lighting', '光照'],
    ['weather', '天气'], ['text', '文字'], ['style_parody', '画风模仿'], ['framing', '取景'],
    ['sexual', '性相关'], ['weapon', '武器'], ['food', '食物'], ['species', '物种'], ['fashion_style', '服装风格'],
  ].map(([value, label]) => [value!, [label!, humanize(value!)]])),
}
export function curationFieldChoices(language: string, field: CurationField) {
  return (field.choices || []).map(value => ({
    title: choiceLabels[field.dest]?.[String(value)]?.[isChinese(language) ? 0 : 1] ?? String(value), value,
  }))
}

type StageCopy = readonly [cnSummary: string, cnNotes: string, enSummary: string, enNotes: string]
export const curationStageCopyCatalog: Record<string, StageCopy> = {
  resize: ['按训练分辨率缩放原图，保留画面比例并建立后续处理工作区。', '读取缩放图的阶段会自动先执行此步骤；分辨率档位应与训练设置一致。', 'Resize source images for training while preserving their aspect ratio.', 'Stages that read resized images run this first. Match the resolution tiers to training.'],
  autotag: ['自动识别图片标签，可补齐、追加或替换现有标注。', '修改写入工作区的修订标注；原标注只作为回退来源，不会在此步骤改写。', 'Fill, extend or replace image captions with automatically detected tags.', 'Changes go to revised workspace captions. Original captions are only a fallback.'],
  position: ['识别多个主体，为各自的外观和位置生成标注子句。', '应用后请重建标注变体与文本缓存；合并位置子句可恢复为平铺标签。', 'Detect multiple subjects and assign their attributes to position clauses.', 'Rebuild caption variants and text caches after applying. Flatten clauses to return to flat tags.'],
  correct: ['整理标签顺序、触发词与画师标记，并按需生成标注变体。', '优先校正工作区修订标注，仅在缺失时读取原标注；不会修改源目录。', 'Organize tag order, trigger words and artist markers, and optionally create caption variants.', 'Correct revised captions, falling back to originals only when needed. Source captions are unchanged.'],
  drop_groups: ['从修订标注中移除指定标签类别，同时保留其余内容和顺序。', '分类表中未知的标签会保留；原标注不会被修改。', 'Remove selected tag categories while preserving the remaining caption and its order.', 'Unknown tags and original source captions are retained.'],
  audit: ['检查单人物标注是否实际包含多个视图或额外人物。', '先查看检测裁剪和对照图，再按判定类型与可信等级应用；应用后需重建文本缓存。', 'Review single-subject captions for repeated views or additional characters.', 'Review crops and sheets before applying allowed verdicts. Rebuild text caches after applying.'],
  ocr: ['识别图片中的对白、文字和拟声词，生成可审阅的文字记录。', '此步骤不修改图片标注。可在导出阶段选择将识别结果附加到标注。', 'Read dialogue, text and sound effects into reviewable OCR records.', 'This does not edit captions. OCR text can be included later during export.'],
  groups: ['按视觉相似度将图片分组，便于检查重复与近似图片。', '分组仅用于浏览和审阅，不会删除图片或改变训练数据。', 'Group visually similar images to review duplicates and near matches.', 'Grouping aids browsing and review; it does not remove images or change training data.'],
  masks_sam: ['使用文本或软提示选择训练区域，生成主体与忽略区域遮罩。', '保留区域参与训练，忽略区域从中排除；没有保留区域时以整图为基础。', 'Create masks for training subjects and ignored areas using text or learned prompts.', 'Keep regions train; ignore regions are removed from them, or from the whole image if none are kept.'],
  masks_merge: ['按相同相对路径合并多个来源的遮罩。', '合并所有需要屏蔽的区域，结果写入工作区的合并遮罩目录。', 'Merge masks from multiple sources by matching their relative paths.', 'Combine all masked-out regions and save them in the workspace mask directory.'],
  export: ['将整理后的图片、标注与遮罩导出到训练数据目录。', '导出可能回写源标注、删除过期附属文件；请先审阅预览。撤销使用任务备份，目标或备份已变化时会拒绝恢复。', 'Publish curated images, captions and masks to the training dataset.', 'Export may update source captions and remove stale sidecars. Review first. Undo uses job backups and refuses recovery if destinations or backups have changed.'],
}
export function curationStageSummary(language: string, stage: CurationStage): string {
  return curationStageCopyCatalog[stage.id]?.[isChinese(language) ? 0 : 2] ?? stage.short ?? stage.title
}
export function curationStageNotes(language: string, stage: CurationStage): string {
  return curationStageCopyCatalog[stage.id]?.[isChinese(language) ? 1 : 3] ?? ''
}
