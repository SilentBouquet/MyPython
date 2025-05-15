document.addEventListener('DOMContentLoaded', function() {
    // 通用元素
    const userAvatar = document.getElementById('user-avatar');
    const userDropdown = document.getElementById('user-dropdown');
    const menuToggle = document.getElementById('menu-toggle');
    const mainNav = document.querySelector('.main-nav');
    const helpLink = document.getElementById('help-link');
    const helpModal = document.getElementById('help-modal');
    const closeHelp = document.getElementById('close-help');
    const toast = document.getElementById('toast');

    // 登录页面元素
    const loginForm = document.getElementById('login-form');

    // 主页面元素
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const dropzone = document.getElementById('dropzone');
    const dropzoneContent = document.getElementById('dropzone-content');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const removeImage = document.getElementById('remove-image');
    const styleCards = document.querySelectorAll('.style-card');
    const convertBtn = document.getElementById('convert-btn');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultDisplay = document.getElementById('result-display');
    const resultImage = document.getElementById('result-image');
    const downloadBtn = document.getElementById('download-btn');
    const shareBtn = document.getElementById('share-btn');
    const processingOverlay = document.getElementById('processing-overlay');
    const processingMessage = document.getElementById('processing-message');
    const progressFill = document.getElementById('progress-fill');

    // 历史记录页面元素
    const historyGrid = document.getElementById('history-grid');
    const emptyHistory = document.getElementById('empty-history');
    const styleFilter = document.getElementById('style-filter');
    const dateSort = document.getElementById('date-sort');
    const imageModal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const modalTitle = document.getElementById('modal-title');
    const closeModal = document.getElementById('close-modal');
    const modalDownload = document.getElementById('modal-download');

    // 用户下拉菜单
    if (userAvatar) {
        userAvatar.addEventListener('click', function(e) {
            e.stopPropagation(); // 阻止事件冒泡
            userDropdown.classList.toggle('active');
        });

        // 点击其他区域关闭下拉菜单
        document.addEventListener('click', function(e) {
            if (userDropdown && userDropdown.classList.contains('active') &&
                !userAvatar.contains(e.target) && !userDropdown.contains(e.target)) {
                userDropdown.classList.remove('active');
            }
        });
    }

    // 移动端菜单
    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            mainNav.classList.toggle('active');
            menuToggle.innerHTML = mainNav.classList.contains('active') ?
                '<i class="fas fa-times"></i>' : '<i class="fas fa-bars"></i>';
        });
    }

    // 帮助模态框
    if (helpLink) {
        helpLink.addEventListener('click', function(e) {
            e.preventDefault();
            helpModal.classList.add('active');
        });
    }

    if (closeHelp) {
        closeHelp.addEventListener('click', function() {
            helpModal.classList.remove('active');
        });

        helpModal.addEventListener('click', function(e) {
            if (e.target === helpModal) {
                helpModal.classList.remove('active');
            }
        });
    }

    // 显示通知消息
    function showToast(message, type = 'success') {
        if (!toast) return; // 防止toast不存在

        const toastIcon = toast.querySelector('.toast-icon i');
        const toastMessage = toast.querySelector('.toast-message');

        toastMessage.textContent = message;

        if (type === 'success') {
            toastIcon.className = 'fas fa-check-circle';
            toastIcon.style.color = 'var(--success-color)';
        } else if (type === 'error') {
            toastIcon.className = 'fas fa-exclamation-circle';
            toastIcon.style.color = 'var(--danger-color)';
        } else if (type === 'info') {
            toastIcon.className = 'fas fa-info-circle';
            toastIcon.style.color = 'var(--accent-blue)';
        }

        toast.classList.add('active');

        setTimeout(() => {
            toast.classList.remove('active');
        }, 3000);
    }

    // === 登录页面 ===
    if (loginForm) {
        const emailInput = document.getElementById('email');
        const passwordInput = document.getElementById('password');
        const emailError = document.getElementById('email-error');
        const passwordError = document.getElementById('password-error');

        // 邮箱格式验证
        emailInput.addEventListener('input', function() {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(this.value) && this.value.length > 0) {
                emailError.textContent = '请输入有效的电子邮箱地址';
            } else {
                emailError.textContent = '';
            }
        });

        // 密码长度验证
        passwordInput.addEventListener('input', function() {
            if (this.value.length > 0 && this.value.length < 8) {
                passwordError.textContent = '密码长度至少为8个字符';
            } else {
                passwordError.textContent = '';
            }
        });

        // 表单提交验证
        loginForm.addEventListener('submit', function(e) {
            let isValid = true;

            // 邮箱验证
            if (emailInput.value.trim() === '') {
                emailError.textContent = '请输入您的电子邮箱';
                isValid = false;
            }

            // 密码验证
            if (passwordInput.value.trim() === '') {
                passwordError.textContent = '请输入您的密码';
                isValid = false;
            } else if (passwordInput.value.length < 8) {
                passwordError.textContent = '密码长度至少为8个字符';
                isValid = false;
            }

            if (!isValid) {
                e.preventDefault();
            }
        });
    }

    // === 图片上传与转换 ===
    if (dropzone) {
        // 修复：使用直接点击打开文件选择器
        dropzone.addEventListener('click', function(e) {
            // 阻止删除按钮点击时触发
            if (e.target.closest('.remove-image')) {
                return;
            }
            imageUpload.click();
        });

        // 文件拖放处理
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            dropzone.classList.add('active');
        }

        function unhighlight() {
            dropzone.classList.remove('active');
        }

        // 处理拖放文件
        dropzone.addEventListener('drop', function(e) {
            const files = e.dataTransfer.files;
            if (files.length) {
                handleFiles(files[0]);
            }
        });

        // 处理文件选择 - 这个是关键部分
        if (imageUpload) {
            imageUpload.addEventListener('change', function(e) {
                console.log('文件已选择'); // 调试信息
                if (this.files && this.files.length) {
                    handleFiles(this.files[0]);
                }
            });
        }

        // 处理文件
        function handleFiles(file) {
            console.log('处理文件:', file.name); // 调试信息

            // 验证文件类型
            const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
            if (!validTypes.includes(file.type)) {
                dropzone.classList.add('error');
                showToast('请上传JPG或PNG格式的图片', 'error');
                setTimeout(() => {
                    dropzone.classList.remove('error');
                }, 500);
                return;
            }

            // 验证文件大小 (5MB限制)
            if (file.size > 5 * 1024 * 1024) {
                dropzone.classList.add('error');
                showToast('图片大小不能超过5MB', 'error');
                setTimeout(() => {
                    dropzone.classList.remove('error');
                }, 500);
                return;
            }

            // 创建预览
            const reader = new FileReader();
            reader.onload = function(e) {
                if (previewImage && previewContainer && dropzoneContent) {
                    previewImage.src = e.target.result;
                    dropzoneContent.style.display = 'none';
                    previewContainer.style.display = 'block';

                    // 预选第一个风格
                    if (styleCards && styleCards.length > 0 && !document.querySelector('.style-card.selected')) {
                        styleCards[0].classList.add('selected');
                    }

                    // 激活转换按钮
                    if (convertBtn) {
                        convertBtn.disabled = false;
                    }
                }
            };
            reader.readAsDataURL(file);
        }

        // 删除预览图片
        if (removeImage) {
            removeImage.addEventListener('click', function(e) {
                e.stopPropagation(); // 阻止事件冒泡以避免触发dropzone点击
                if (imageUpload) {
                    imageUpload.value = '';
                }
                if (previewContainer && dropzoneContent) {
                    previewContainer.style.display = 'none';
                    dropzoneContent.style.display = 'block';
                }
                if (convertBtn) {
                    convertBtn.disabled = true;
                }
            });
        }

        // 风格选择
        if (styleCards) {
            styleCards.forEach(card => {
                card.addEventListener('click', function() {
                    styleCards.forEach(c => c.classList.remove('selected'));
                    this.classList.add('selected');
                });
            });
        }

        // 转换过程
        if (convertBtn) {
            convertBtn.addEventListener('click', function() {
                if (!imageUpload || !imageUpload.files || imageUpload.files.length === 0) {
                    showToast('请先上传图片', 'error');
                    return;
                }

                // 获取选中的风格
                const selectedStyle = document.querySelector('.style-card.selected');
                if (!selectedStyle) {
                    showToast('请选择一种风格', 'error');
                    return;
                }

                const style = selectedStyle.dataset.style;

                // 创建FormData对象
                const formData = new FormData();
                formData.append('file', imageUpload.files[0]);
                formData.append('style', style);

                // 显示处理中状态
                if (processingOverlay) {
                    processingOverlay.classList.add('active');
                }

                let messages = {
                    'charcoal': '正在创建素描风格...',
                    'watercolor': '正在绘制水彩效果...',
                    'impression': '正在以印象派风格创作...'
                };

                if (processingMessage) {
                    processingMessage.textContent = messages[style];
                }

                // 模拟进度条
                let progress = 0;
                const progressInterval = setInterval(() => {
                    if (progress >= 90) {
                        clearInterval(progressInterval);
                    } else {
                        progress += 5;
                        if (progressFill) {
                            progressFill.style.width = `${progress}%`;
                        }
                    }
                }, 300);

                // 发送转换请求
                fetch('/convert', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('转换失败');
                    }
                    return response.json();
                })
                .then(data => {
                    clearInterval(progressInterval);
                    if (progressFill) {
                        progressFill.style.width = '100%';
                    }

                    setTimeout(() => {
                        if (processingOverlay) {
                            processingOverlay.classList.remove('active');
                        }
                        if (progressFill) {
                            progressFill.style.width = '0';
                        }

                        if (data.success) {
                            // 显示结果
                            if (resultImage && resultPlaceholder && resultDisplay) {
                                resultImage.src = data.result_url;
                                resultPlaceholder.style.display = 'none';
                                resultDisplay.style.display = 'block';
                            }

                            // 设置下载链接
                            if (downloadBtn) {
                                downloadBtn.onclick = function() {
                                    const link = document.createElement('a');
                                    link.href = data.result_url;
                                    link.download = `${style}_result.jpg`;
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                };
                            }

                            // 分享按钮
                            if (shareBtn) {
                                shareBtn.onclick = function() {
                                    if (navigator.share) {
                                        navigator.share({
                                            title: '我的艺术风格转换',
                                            text: `查看我用${messages[style].replace('正在', '').replace('...', '')}转换的图片！`,
                                            url: window.location.origin + data.result_url
                                        })
                                        .catch(error => {
                                            console.log('分享失败:', error);
                                        });
                                    } else {
                                        showToast('您的浏览器不支持分享功能', 'info');
                                    }
                                };
                            }

                            showToast('风格转换成功！', 'success');
                        } else {
                            // 转换失败
                            showToast(data.error || '转换失败，请重试', 'error');
                            if (resultPlaceholder && resultDisplay) {
                                resultPlaceholder.style.display = 'block';
                                resultDisplay.style.display = 'none';
                            }
                        }
                    }, 500);
                })
                .catch(error => {
                    clearInterval(progressInterval);
                    if (processingOverlay) {
                        processingOverlay.classList.remove('active');
                    }
                    if (progressFill) {
                        progressFill.style.width = '0';
                    }
                    showToast('转换过程发生错误：' + error.message, 'error');
                    console.error('Error:', error);
                });
            });
        }
    }

    // === 历史记录页面 ===
    if (historyGrid) {
        let historyData = [];

        // 获取历史记录
        function fetchHistory() {
            fetch('/api/history')
                .then(response => response.json())
                .then(data => {
                    historyData = data;

                    if (historyData.length === 0) {
                        historyGrid.innerHTML = '';
                        emptyHistory.style.display = 'flex';
                    } else {
                        emptyHistory.style.display = 'none';
                        renderHistory(historyData);
                    }
                })
                .catch(error => {
                    console.error('Error fetching history:', error);
                    historyGrid.innerHTML = `
                        <div class="error-message" style="grid-column: 1/-1; text-align: center; padding: 30px;">
                            <i class="fas fa-exclamation-circle" style="font-size: 2rem; color: var(--danger-color); margin-bottom: 15px;"></i>
                            <p>获取历史记录失败，请刷新页面重试</p>
                        </div>
                    `;
                });
        }

        // 渲染历史记录
        function renderHistory(data) {
            historyGrid.innerHTML = '';

            if (data.length === 0) {
                historyGrid.innerHTML = `
                    <div class="no-results" style="grid-column: 1/-1; text-align: center; padding: 30px;">
                        <p>没有符合条件的记录</p>
                    </div>
                `;
                return;
            }

            data.forEach((item, index) => {
                let styleText = '';
                let styleClass = '';

                if (item.style === 'charcoal') {
                    styleText = '素描风格';
                    styleClass = 'charcoal';
                } else if (item.style === 'watercolor') {
                    styleText = '水彩风格';
                    styleClass = 'watercolor';
                } else if (item.style === 'impression') {
                    styleText = '印象派风格';
                    styleClass = 'impression';
                }

                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.dataset.id = index;
                historyItem.dataset.style = item.style;

                historyItem.innerHTML = `
                    <div class="history-item-header">
                        <div class="history-date">${item.date}</div>
                        <div class="history-style ${styleClass}">${styleText}</div>
                    </div>
                    <div class="history-images">
                        <div class="history-image">
                            <img src="${item.original}" alt="原始图片" data-src="${item.original}" data-type="original">
                            <div class="image-label">原始图片</div>
                        </div>
                        <div class="history-image">
                            <img src="${item.result}" alt="转换结果" data-src="${item.result}" data-type="result">
                            <div class="image-label">转换结果</div>
                        </div>
                    </div>
                    <div class="history-actions">
                        <button class="download-action" data-url="${item.result}">
                            <i class="fas fa-download"></i> 下载
                        </button>
                        <button class="delete-action" data-id="${index}">
                            <i class="fas fa-trash"></i> 删除
                        </button>
                    </div>
                `;

                historyGrid.appendChild(historyItem);
            });

            // 添加下载和删除事件
            document.querySelectorAll('.download-action').forEach(button => {
                button.addEventListener('click', function() {
                    const url = this.dataset.url;
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = url.split('/').pop();
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    showToast('图片下载中...', 'info');
                });
            });

            document.querySelectorAll('.delete-action').forEach(button => {
                button.addEventListener('click', function() {
                    const id = this.dataset.id;

                    if (confirm('确定要删除这条记录吗？')) {
                        fetch(`/api/history/${id}`, {
                            method: 'DELETE'
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                const item = document.querySelector(`.history-item[data-id="${id}"]`);
                                item.style.animation = 'fadeOut 0.3s forwards';

                                setTimeout(() => {
                                    item.remove();
                                    historyData = historyData.filter((_, index) => index != id);

                                    if (historyData.length === 0) {
                                        emptyHistory.style.display = 'flex';
                                    }

                                    showToast('记录已删除', 'success');
                                }, 300);
                            } else {
                                showToast('删除失败，请重试', 'error');
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            showToast('删除过程发生错误', 'error');
                        });
                    }
                });
            });

            // 图片预览
            document.querySelectorAll('.history-image img').forEach(img => {
                img.addEventListener('click', function() {
                    const src = this.dataset.src;
                    const type = this.dataset.type;

                    modalImage.src = src;
                    modalTitle.textContent = type === 'original' ? '原始图片' : '转换结果';
                    modalDownload.onclick = function() {
                        const link = document.createElement('a');
                        link.href = src;
                        link.download = src.split('/').pop();
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    };

                    imageModal.classList.add('active');
                });
            });
        }

        // 筛选与排序
        function filterAndSortHistory() {
            const style = styleFilter.value;
            const sort = dateSort.value;

            let filteredData = [...historyData];

            // 应用风格筛选
            if (style !== 'all') {
                filteredData = filteredData.filter(item => item.style === style);
            }

            // 应用日期排序
            filteredData.sort((a, b) => {
                const dateA = new Date(a.date);
                const dateB = new Date(b.date);

                if (sort === 'newest') {
                    return dateB - dateA;
                } else {
                    return dateA - dateB;
                }
            });

            renderHistory(filteredData);
        }

        // 监听筛选和排序变化
        if (styleFilter && dateSort) {
            styleFilter.addEventListener('change', filterAndSortHistory);
            dateSort.addEventListener('change', filterAndSortHistory);
        }

        // 关闭模态框
        if (closeModal) {
            closeModal.addEventListener('click', function() {
                imageModal.classList.remove('active');
            });

            imageModal.addEventListener('click', function(e) {
                if (e.target === imageModal || e.target.classList.contains('modal-overlay')) {
                    imageModal.classList.remove('active');
                }
            });
        }

        // 初始化加载历史记录
        fetchHistory();
    }

    // 添加淡出动画
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.9); }
    }

    function loadCharts() {
        fetch('/api/stats')
            .then(response => response.json())
            .then(stats => {
                try {
                    // 尝试使用Chart.js渲染
                    renderStyleChart(stats);
                    // ... 其他渲染代码 ...
                } catch (error) {
                    console.error('Chart.js渲染失败，使用备用方案:', error);
                    // 使用备用方案
                    renderSimpleCharts(stats);
                }
            })
            .catch(error => {
                console.error('Error loading stats:', error);
            });
    }
});