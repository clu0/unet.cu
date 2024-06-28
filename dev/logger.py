from collections import defaultdict


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            self.file = filename_or_file
            self.own_file = False

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writekvs(self, kvs):
        key2str = {}
        for key, val in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        if len(key2str) == 0:
            print("WARNING: tried to write empty k-v dict")
            return
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # write data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for key, val in sorted(key2str.items()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        self.file.flush()

    def writeseq(self, seq):
        seq = list(seq)
        for i, elem in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:
                self.file.write(" ")
            self.file.write("\n")
            self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "a+t")

    def writekvs(self, kvs):
        keys = sorted(kvs.keys())
        if self.file.tell() == 0:
            # write header
            self.file.write(",".join(keys) + "\n")

        # write values
        self.file.write(",".join(str(kvs[key]) for key in keys) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class Logger(object):
    def __init__(self, output_formats):
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)
        self.output_formats = output_formats

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        old_val, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = old_val * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] += 1

    def dumpkvs(self):
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                if isinstance(args, str):
                    fmt.writeseq([args])
                else:
                    fmt.writeseq(map(str, args))

    def close(self):
        for fmt in self.output_formats:
            fmt.close()
