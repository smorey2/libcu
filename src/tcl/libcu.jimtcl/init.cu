// autogenerated - do not edit
#include "jim.h"

__device__ int Jim_bootstrapInit(Jim_Interp *interp) {
	if (Jim_PackageProvide(interp, "bootstrap", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	return Jim_EvalSource(interp, "bootstrap.tcl", 1,
		"proc package {args} {}\n"
		);
}

__device__ int Jim_initjimshInit(Jim_Interp *interp) {
	if (Jim_PackageProvide(interp, "initjimsh", "1.0", JIM_ERRMSG)) return JIM_ERROR;
	return Jim_EvalSource(interp, "initjimsh.tcl", 1,
		"proc _jimsh_init {} {\n"
		"	rename _jimsh_init {}\n"
		"	global jim::exe jim::argv0 tcl_interactive auto_path tcl_platform\n"
		"\n"
		"	# Stash the result of [info nameofexecutable] now, before a possible [cd]\n"
		"	if {[exists jim::argv0]} {\n"
		"		if {[string match \"*/*\" $jim::argv0]} {\n"
		"			set jim::exe [file join [pwd] $jim::argv0]\n"
		"		} else {\n"
		"			foreach path [split [env PATH \"\"] $tcl_platform(pathSeparator)] {\n"
		"				set exec [file join [pwd] [string map {\\\\ /} $path] $jim::argv0]\n"
		"				if {[file executable $exec]} {\n"
		"					set jim::exe $exec\n"
		"					break\n"
		"				}\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"\n"
		"	# Add to the standard auto_path\n"
		"	lappend p {*}[split [env JIMLIB {}] $tcl_platform(pathSeparator)]\n"
		"	if {[exists jim::exe]} {\n"
		"		lappend p [file dirname $jim::exe]\n"
		"	}\n"
		"	lappend p {*}$auto_path\n"
		"	set auto_path $p\n"
		"\n"
		"	if {[env HOME {}] ne \"\"} {\n"
		"		foreach src {.jimrc jimrc.tcl} {\n"
		"			if {[file exists [env HOME]/$src]} {\n"
		"				uplevel #0 source [env HOME]/$src\n"
		"				break\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"	return \"\"\n"
		"}\n"
		"\n"
		"if {$tcl_platform(platform) eq \"windows\"} {\n"
		"	set jim::argv0 [string map {\\\\ /} $jim::argv0]\n"
		"}\n"
		"\n"
		"_jimsh_init\n"
		//"cd tests; source array.test;\n"
		);
}

__device__ int Jim_globInit(Jim_Interp *interp) {
	if (Jim_PackageProvide(interp, "glob", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	return Jim_EvalSource(interp, "glob.tcl", 1,
		"package require readdir\n"
		"\n"
		"proc glob.globdir {dir pattern} {\n"
		"	if {[file exists $dir/$pattern]} {\n"
		"		# Simple case\n"
		"		return [list $pattern]\n"
		"	}\n"
		"	set result {}\n"
		"	set files [readdir $dir]\n"
		"	lappend files . ..\n"
		"\n"
		"	foreach name $files {\n"
		"		if {[string match $pattern $name]} {\n"
		"			if {[string index $name 0] eq \".\" && [string index $pattern 0] ne \".\"} {\n"
		"				continue\n"
		"			}\n"
		"			lappend result $name\n"
		"		}\n"
		"	}\n"
		"	return $result\n"
		"}\n"
		"\n"
		"proc glob.explode {pattern} {\n"
		"	set oldexp {}\n"
		"	set newexp {\"\"}\n"
		"\n"
		"	while 1 {\n"
		"		set oldexp $newexp\n"
		"		set newexp {}\n"
		"		set ob [string first \\{ $pattern]\n"
		"		set cb [string first \\} $pattern]\n"
		"\n"
		"		if {$ob < $cb && $ob != -1} {\n"
		"			set mid [string range $pattern 0 $ob-1]\n"
		"			set subexp [lassign [glob.explode [string range $pattern $ob+1 end]] pattern]\n"
		"			if {$pattern eq \"\"} {\n"
		"				error \"unmatched open brace in glob pattern\"\n"
		"			}\n"
		"			set pattern [string range $pattern 1 end]\n"
		"\n"
		"			foreach subs $subexp {\n"
		"				foreach sub [split $subs ,] {\n"
		"					foreach old $oldexp {\n"
		"						lappend newexp $old$mid$sub\n"
		"					}\n"
		"				}\n"
		"			}\n"
		"		} elseif {$cb != -1} {\n"
		"			set suf  [string range $pattern 0 $cb-1]\n"
		"			set rest [string range $pattern $cb end]\n"
		"			break\n"
		"		} else {\n"
		"			set suf  $pattern\n"
		"			set rest \"\"\n"
		"			break\n"
		"		}\n"
		"	}\n"
		"\n"
		"	foreach old $oldexp {\n"
		"		lappend newexp $old$suf\n"
		"	}\n"
		"	list $rest {*}$newexp\n"
		"}\n"
		"\n"
		"proc glob.glob {base pattern} {\n"
		"	set dir [file dirname $pattern]\n"
		"	if {$pattern eq $dir || $pattern eq \"\"} {\n"
		"		return [list [file join $base $dir] $pattern]\n"
		"	} elseif {$pattern eq [file tail $pattern]} {\n"
		"		set dir \"\"\n"
		"	}\n"
		"\n"
		"	set dirlist [glob.glob $base $dir]\n"
		"	set pattern [file tail $pattern]\n"
		"\n"
		"	set result {}\n"
		"	foreach {realdir dir} $dirlist {\n"
		"		if {![file isdir $realdir]} {\n"
		"			continue\n"
		"		}\n"
		"		if {[string index $dir end] ne \"/\" && $dir ne \"\"} {\n"
		"			append dir /\n"
		"		}\n"
		"		foreach name [glob.globdir $realdir $pattern] {\n"
		"			lappend result [file join $realdir $name] $dir$name\n"
		"		}\n"
		"	}\n"
		"	return $result\n"
		"}\n"
		"\n"
		"proc glob {args} {\n"
		"	set nocomplain 0\n"
		"	set base \"\"\n"
		"	set tails 0\n"
		"\n"
		"	set n 0\n"
		"	foreach arg $args {\n"
		"		if {[info exists param]} {\n"
		"			set $param $arg\n"
		"			unset param\n"
		"			incr n\n"
		"			continue\n"
		"		}\n"
		"		switch -glob -- $arg {\n"
		"			-d* {\n"
		"				set switch $arg\n"
		"				set param base\n"
		"			}\n"
		"			-n* {\n"
		"				set nocomplain 1\n"
		"			}\n"
		"			-t* {\n"
		"				set tails 1"
		"			}\n"
		"			-- {\n"
		"				incr n\n"
		"				break\n"
		"			}\n"
		"			-* {\n"
		"				return -code error \"bad option \\\"$switch\\\": must be -directory, -nocomplain, -tails, or --\"\n"
		"			}\n"
		"			* {\n"
		"				break\n"
		"			}\n"
		"		}\n"
		"		incr n\n"
		"	}\n"
		"	if {[info exists param]} {\n"
		"		return -code error \"missing argument to \\\"$switch\\\"\"\n"
		"	}\n"
		"	if {[llength $args] <= $n} {\n"
		"		return -code error \"wrong # args: should be \\\"glob ?options? pattern ?pattern ...?\\\"\"\n"
		"	}\n"
		"\n"
		"	set args [lrange $args $n end]\n"
		"\n"
		"	set result {}\n"
		"	foreach pattern $args {\n"
		"		set pattern [string map {\n"
		"			\\\\\\\\ \\x01 \\\\\\{ \\x02 \\\\\\} \\x03 \\\\, \\x04\n"
		"		} $pattern]\n"
		"		set patexps [lassign [glob.explode $pattern] rest]\n"
		"		if {$rest ne \"\"} {\n"
		"			return -code error \"unmatched close brace in glob pattern\"\n"
		"		}\n"
		"		foreach patexp $patexps {\n"
		"			set patexp [string map {\n"
		"				\\x01 \\\\\\\\ \\x02 \\{ \\x03 \\} \\x04 ,\n"
		"			} $patexp]\n"
		"			foreach {realname name} [glob.glob $base $patexp] {\n"
		"				incr n\n"
		"				if {$tails} {\n"
		"					lappend result $name\n"
		"				} else {\n"
		"					lappend result [file join $base $name]\n"
		"				}\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"\n"
		"	if {!$nocomplain && [llength $result] == 0} {\n"
		"		set s $(([llength $args] > 1) ? \"s\" : \"\")\b"
		"		return -code error \"no files matched glob pattern$s \\\"[join $args]\\\"\"\n"
		"	}\n"
		"\n"
		"	return $result\n"
		"}\n"
		);
}

__device__ int Jim_stdlibInit(Jim_Interp *interp) {
	if (Jim_PackageProvide(interp, "stdlib", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	return Jim_EvalSource(interp, "stdlib.tcl", 1,
		"\n"
		"proc lambda {arglist args} {\n"
		"	tailcall proc [ref {} function lambda.finalizer] $arglist {*}$args\n"
		"}\n"
		"\n"
		"proc lambda.finalizer {name val} {\n"
		"	rename $name {}\n"
		"}\n"
		"\n"
		"proc curry {args} {\n"
		"	alias [ref {} function lambda.finalizer] {*}$args\n"
		"}\n"
		"\n"
		"proc function {value} {\n"
		"	return $value\n"
		"}\n"
		"\n"
		"proc stacktrace {} {\n"
		"	set trace {}\n"
		"	foreach level [range 1 [info level]] {\n"
		"		lassign [info frame -$level] p f l\n"
		"		lappend trace $p $f $l\n"
		"	}\n"
		"	return $trace\n"
		"}\n"
		"\n"
		"proc stackdump {stacktrace} {\n"
		"	set result {}\n"
		"	set count 0\n"
		"	foreach {l f p} [lreverse $stacktrace] {\n"
		"		if {$count} {\n"
		"			append result \\n\n"
		"		}\n"
		"		incr count\n"
		"		if {$p ne \"\"} {\n"
		"			append result \"in procedure '$p' \"\n"
		"			if {$f ne \"\"} {\n"
		"				append result \"called \"\n"
		"			}\n"
		"		}\n"
		"		if {$f ne \"\"} {\n"
		"			append result \"at file \\\"$f\\\", line $l\"\n"
		"		}\n"
		"	}\n"
		"	return $result\n"
		"}\n"
		"\n"
		"proc errorInfo {msg {stacktrace \"\"}} {\n"
		"	if {$stacktrace eq \"\"} {\n"
		"		set stacktrace [info stacktrace]\n"
		"	}\n"
		"	lassign $stacktrace p f l\n"
		"	if {$f ne \"\"} {\n"
		"		set result \"Runtime Error: $f:$l: \"\n"
		"	}\n"
		"	append result \"$msg\\n\"\n"
		"	append result [stackdump $stacktrace]\n"
		"\n"
		"\n"
		"	string trim $result\n"
		"}\n"
		"\n"
		"proc {info nameofexecutable} {} {\n"
		"	if {[info exists ::jim_argv0]} {\n"
		"		if {[string match \"*/*\" $::jim_argv0]} {\n"
		"			return [file join [pwd] $::jim_argv0]\n"
		"		}\n"
		"		foreach path [split [env PATH \"\"] $::tcl_platform(pathSeparator)] {\n"
		"			set exec [file join [pwd] [string map {\\\\ /} $path] $::jim_argv0]\n"
		"			if {[file executable $exec]} {\n"
		"				return $exec\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"	return \"\"\n"
		"}\n"
		"\n"
		"proc {dict with} {dictVar args script} {\n"
		"	upvar $dictVar dict\n"
		"	set keys {}\n"
		"	foreach {n v} [dict get $dict {*}$args] {\n"
		"		upvar $n var_$n\n"
		"		set var_$n $v\n"
		"		lappend keys $n\n"
		"	}\n"
		"	catch {uplevel 1 $script} msg opts\n"
		"	if {[info exists dict] && [dict exists $dict {*}$args]} {\n"
		"		foreach n $keys {\n"
		"			if {[info exists var_$n]} {\n"
		"				dict set dict {*}$args $n [set var_$n]\n"
		"			} else {\n"
		"				dict unset dict {*}$args $n\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"	return {*}$opts $msg\n"
		"}\n"
		"\n"
		"proc {dict merge} {dict args} {\n"
		"	foreach d $args {\n"
		"\n"
		"		dict size $d\n"
		"		foreach {k v} $d {\n"
		"			dict set dict $k $v\n"
		"		}\n"
		"	}\n"
		"	return $dict\n"
		"}\n"
		);
}

__device__ int Jim_tclcompatInit(Jim_Interp *interp) {
	if (Jim_PackageProvide(interp, "tclcompat", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	return Jim_EvalSource(interp, "tclcompat.tcl", 1,
		"set env [env]\n"
		"\n"
		"if {[info commands stdout] ne \"\"} {\n"
		"\n"
		"	foreach p {gets flush close eof seek tell} {\n"
		"		proc $p {chan args} {p} {\n"
		"			tailcall $chan $p {*}$args\n"
		"		}\n"
		"	}\n"
		"	unset p\n"
		"\n"
		"	proc puts {{-nonewline {}} {chan stdout} msg} {\n"
		"		if {${-nonewline} ni {-nonewline {}}} {\n"
		"			tailcall ${-nonewline} puts $msg\n"
		"		}\n"
		"		tailcall $chan puts {*}${-nonewline} $msg\n"
		"	}\n"
		"\n"
		"	proc read {{-nonewline {}} chan} {\n"
		"		if {${-nonewline} ni {-nonewline {}}} {\n"
		"			tailcall ${-nonewline} read {*}${chan}\n"
		"		}\n"
		"		tailcall $chan read {*}${-nonewline}\n"
		"	}\n"
		"\n"
		"	proc fconfigure {f args} {\n"
		"		foreach {n v} $args {\n"
		"			switch -glob -- $n {\n"
		"				-bl* {\n"
		"					$f ndelay $(!$v)\n"
		"				}\n"
		"				-bu* {\n"
		"					$f buffering $v\n"
		"				}\n"
		"				-tr* {\n"
		"\n"
		"				}\n"
		"				default {\n"
		"					return -code error \"fconfigure: unknown option $n\"\n"
		"				}\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"}\n"
		"\n"
		"proc case {var args} {\n"
		"\n"
		"	if {[lindex $args 0] eq \"in\"} {\n"
		"		set args [lrange $args 1 end]\n"
		"	}\n"
		"\n"
		"	if {[llength $args] == 1} {\n"
		"		set args [lindex $args 0]\n"
		"	}\n"
		"\n"
		"	if {[llength $args] % 2 != 0} {\n"
		"		return -code error \"extra case pattern with no body\"\n"
		"	}\n"
		"\n"
		"	local proc case.checker {value pattern} {\n"
		"		string match $pattern $value\n"
		"	}\n"
		"\n"
		"	foreach {value action} $args {\n"
		"		if {$value eq \"default\"} {\n"
		"			set do_action $action\n"
		"			continue\n"
		"		} elseif {[lsearch -bool -command case.checker $value $var]} {\n"
		"			set do_action $action\n"
		"			break\n"
		"		}\n"
		"	}\n"
		"\n"
		"	if {[info exists do_action]} {\n"
		"		set rc [catch [list uplevel 1 $do_action] result opts]\n"
		"		if {$rc} {\n"
		"			incr opts(-level)\n"
		"		}\n"
		"		return {*}$opts $result\n"
		"	}\n"
		"}\n"
		"\n"
		"proc fileevent {args} {\n"
		"	tailcall {*}$args\n"
		"}\n"
		"\n"
		"proc parray {arrayname {pattern *} {puts puts}} {\n"
		"	upvar $arrayname a\n"
		"\n"
		"	set max 0\n"
		"	foreach name [array names a $pattern]] {\n"
		"		if {[string length $name] > $max} {\n"
		"			set max [string length $name]\n"
		"		}\n"
		"	}\n"
		"	incr max [string length $arrayname]\n"
		"	incr max 2\n"
		"	foreach name [lsort [array names a $pattern]] {\n"
		"		$puts [format \"%-${max}s = %s\" $arrayname\\($name\\) $a($name)]\n"
		"	}\n"
		"}\n"
		"\n"
		"proc {file copy} {{force {}} source target} {\n"
		"	try {\n"
		"		if {$force ni {{} -force}} {\n"
		"			error \"bad option \\\"$force\\\": should be -force\"\n"
		"		}\n"
		"\n"
		"		set in [open $source]\n"
		"\n"
		"		if {$force eq \"\" && [file exists $target]} {\n"
		"			$in close\n"
		"			error \"error copying \\\"$source\\\" to \\\"$target\\\": file already exists\"\n"
		"		}\n"
		"		set out [open $target w]\n"
		"		$in copyto $out\n"
		"		$out close\n"
		"	} on error {msg opts} {\n"
		"		incr opts(-level)\n"
		"		return {*}$opts $msg\n"
		"	} finally {\n"
		"		catch {$in close}\n"
		"	}\n"
		"}\n"
		"\n"
		"proc popen {cmd {mode r}} {\n"
		"	lassign [socket pipe] r w\n"
		"	try {\n"
		"		if {[string match \"w*\" $mode]} {\n"
		"			lappend cmd <@$r &\n"
		"			set pids [exec {*}$cmd]\n"
		"			$r close\n"
		"			set f $w\n"
		"		} else {\n"
		"			lappend cmd >@$w &\n"
		"			set pids [exec {*}$cmd]\n"
		"			$w close\n"
		"			set f $r\n"
		"		}\n"
		"		lambda {cmd args} {f pids} {\n"
		"			if {$cmd eq \"pid\"} {\n"
		"				return $pids\n"
		"			}\n"
		"			if {$cmd eq \"close\"} {\n"
		"				$f close\n"
		"\n"
		"				foreach p $pids { os.wait $p }\n"
		"				return\n"
		"			}\n"
		"			tailcall $f $cmd {*}$args\n"
		"		}\n"
		"	} on error {error opts} {\n"
		"		$r close\n"
		"		$w close\n"
		"		error $error\n"
		"	}\n"
		"}\n"
		"\n"
		"local proc pid {{chan {}}} {\n"
		"	if {$chan eq \"\"} {\n"
		"		tailcall upcall pid\n"
		"	}\n"
		"	if {[catch {$chan tell}]} {\n"
		"		return -code error \"can not find channel named \\\"$chan\\\"\"\n"
		"	}\n"
		"	if {[catch {$chan pid} pids]} {\n"
		"		return \"\"\n"
		"	}\n"
		"	return $pids\n"
		"}\n"
		"\n"
		"proc try {args} {\n"
		"	set catchopts {}\n"
		"	while {[string match -* [lindex $args 0]]} {\n"
		"		set args [lassign $args opt]\n"
		"		if {$opt eq \"--\"} {\n"
		"			break\n"
		"		}\n"
		"		lappend catchopts $opt\n"
		"	}\n"
		"	if {[llength $args] == 0} {\n"
		"		return -code error {wrong # args: should be \"try ?options? script ?argument ...?\"}\n"
		"	}\n"
		"	set args [lassign $args script]\n"
		"	set code [catch -eval {*}$catchopts [list uplevel 1 $script] msg opts]\n"
		"\n"
		"	set handled 0\n"
		"\n"
		"	foreach {on codes vars script} $args {\n"
		"		switch -- $on \\\n"
		"			on {\n"
		"				if {!$handled && ($codes eq \"*\" || [info returncode $code] in $codes)} {\n"
		"					lassign $vars msgvar optsvar\n"
		"					if {$msgvar ne \"\"} {\n"
		"						upvar $msgvar hmsg\n"
		"						set hmsg $msg\n"
		"					}\n"
		"					if {$optsvar ne \"\"} {\n"
		"						upvar $optsvar hopts\n"
		"						set hopts $opts\n"
		"					}\n"
		"\n"
		"					set code [catch [list uplevel 1 $script] msg opts]\n"
		"					incr handled\n"
		"				}\n"
		"			} \\\n"
		"			finally {\n"
		"				set finalcode [catch [list uplevel 1 $codes] finalmsg finalopts]\n"
		"				if {$finalcode} {\n"
		"\n"
		"					set code $finalcode\n"
		"					set msg $finalmsg\n"
		"					set opts $finalopts\n"
		"				}\n"
		"				break\n"
		"			} \\\n"
		"			default {\n"
		"				return -code error \"try: expected 'on' or 'finally', got '$on'\"\n"
		"			}\n"
		"	}\n"
		"\n"
		"	if {$code} {\n"
		"		incr opts(-level)\n"
		"		return {*}$opts $msg\n"
		"	}\n"
		"	return $msg\n"
		"}\n"
		"\n"
		"proc throw {code {msg \"\"}} {\n"
		"	return -code $code $msg\n"
		"}\n"
		"\n"
		"proc {file delete force} {path} {\n"
		"	foreach e [readdir $path] {\n"
		"		file delete -force $path/$e\n"
		"	}\n"
		"	file delete $path\n"
		"}\n"
		);
}
