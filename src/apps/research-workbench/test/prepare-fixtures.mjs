import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=path.join(path.dirname(fileURLToPath(import.meta.url)),'fixtures');await fs.mkdir(dir,{recursive:true});
await fs.writeFile(path.join(dir,'editor.md'),'# Editor fixture\n\nOriginal content.\n');
execFileSync('ffmpeg',['-v','error','-f','lavfi','-i','color=c=0xc5eb98:s=320x180:d=2','-c:v','libvpx','-b:v','80k','-y',path.join(dir,'preview.webm')]);
execFileSync('ffmpeg',['-v','error','-f','lavfi','-i','color=c=0xc5eb98:s=320x180','-frames:v','1','-y',path.join(dir,'preview.png')]);
